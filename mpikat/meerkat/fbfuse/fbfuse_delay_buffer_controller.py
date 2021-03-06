"""
Copyright (c) 2018 Ewan Barr <ebarr@mpifr-bonn.mpg.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import logging
import json
import posix_ipc
import time
import socket
import os
import struct
import ctypes as C
import numpy as np
from collections import OrderedDict
from mmap import mmap
from threading import Thread, Event
from tornado.gen import coroutine
from tornado.ioloop import PeriodicCallback
from katpoint import Antenna, Target
from mosaic import DelayPolynomial
from mpikat.core.utils import Timer

log = logging.getLogger("mpikat.fbfuse_delay_buffer_controller")

DEFAULT_TARGET = "unset,radec,0,0"
DEFAULT_UPDATE_RATE = 2.0
DEFAULT_DELAY_SPAN = 2 * DEFAULT_UPDATE_RATE
CONTROL_SOCKET_ADDR = "/tmp/fbfuse_control.sock"


def delay_model_type(nbeams, nants):
    class DelayModel(C.LittleEndianStructure):
        _fields_ = [
            ("epoch", C.c_double),
            ("duration", C.c_double),
            ("delays", C.c_float * (2 * nbeams * nants))
        ]
    return DelayModel


class OfflineControlThread(Thread):

    def __init__(self, controller):
        Thread.__init__(self)
        self.daemon = True
        self._controller = controller
        self._control_socket = None
        self._stop_event = Event()
        self._ready_event = Event()

    def start_control_socket(self):
        if self._control_socket:
            self.stop_control_socket()
        log.debug("Starting up control socket")
        self._control_socket = socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            os.unlink(CONTROL_SOCKET_ADDR)
        except OSError:
            if os.path.exists(CONTROL_SOCKET_ADDR):
                raise
        self._control_socket.bind(CONTROL_SOCKET_ADDR)
        self._control_socket.listen(1)
        self._control_socket.settimeout(1)
        self._ready_event.set()
        log.debug("Control socket running")

    def stop_control_socket(self):
        log.debug("Closing control socket")
        self._control_socket.close()

    def accept(self):
        while not self._stop_event.is_set():
            try:
                conn, addr = self._control_socket.accept()
            except socket.timeout:
                continue
            else:
                log.debug("Handling control socket connection")
                self.handle(conn)

    def handle(self, conn):
        conn.settimeout(1)
        log.debug("Receiving delay epoch")
        epoch = struct.unpack("d", conn.recv(8))[0]
        log.debug("Delay epoch = {}".format(epoch))
        try:
            self._controller.update_delays(epoch=epoch)
        except Exception:
            log.exception("Error on delay update")
            conn.sendall(struct.pack("b", 0))
        else:
            conn.sendall(struct.pack("b", 1))
        conn.close()

    def run(self):
        self.start_control_socket()
        try:
            self.accept()
        except Exception:
            log.exception("Error in control socket thread")
        finally:
            self.stop_control_socket()

    def stop(self):
        self._stop_event.set()

    def wait_until_ready(self, timeout=2):
        self._ready_event.wait(timeout)


class DelayBufferController(object):

    def __init__(self, delay_client, ordered_beams, ordered_antennas,
                 nreaders, offline=False):
        """
        @brief    Controls shared memory delay buffers that are accessed by one or more
                  beamformer instances.

        @params   delay_client       A KATCPResourceClient connected to an FBFUSE delay engine server
        @params   ordered_beams      A list of beam IDs in the order that they should be generated by the beamformer
        @params   orderded_antennas  A list of antenna IDs in the order which they should be captured by the beamformer
        @params   nreaders           The number of posix shared memory readers that will access the memory
                                     buffers that are managed by this instance.


        @note     The delay model definition in FBFUSE looks like:
                  @code
                      struct DelayModel
                      {
                          double epoch;
                          double duration;
                          float2 delays[FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS];
                      };
                  @endcode
        """
        self._nreaders = nreaders
        self._delay_client = delay_client
        self._ordered_antennas = ordered_antennas
        self._ordered_beams = ordered_beams
        self.shared_buffer_key = "delay_buffer"
        self.mutex_semaphore_key = "delay_buffer_mutex"
        self.counting_semaphore_key = "delay_buffer_count"
        self._nbeams = len(self._ordered_beams)
        self._nantennas = len(self._ordered_antennas)
        self._delay_model = delay_model_type(
            self._nbeams, self._nantennas)()
        self._targets = OrderedDict()
        for beam in self._ordered_beams:
            self._targets[beam] = Target(DEFAULT_TARGET)
        self._phase_reference = Target(DEFAULT_TARGET)
        self._update_rate = DEFAULT_UPDATE_RATE
        self._delay_span = DEFAULT_DELAY_SPAN
        self._update_callback = None
        self._beam_callbacks = {}
        self._offline = offline
        self._offline_control_thread = None

    def unlink_all(self):
        """
        @brief   Unlink (remove) all posix shared memory sections and semaphores.
        """
        log.debug(
            "Unlinking all relevant posix shared memory segments and semaphores")
        try:
            posix_ipc.unlink_semaphore(self.counting_semaphore_key)
        except posix_ipc.ExistentialError:
            pass
        try:
            posix_ipc.unlink_semaphore(self.mutex_semaphore_key)
        except posix_ipc.ExistentialError:
            pass
        try:
            posix_ipc.unlink_shared_memory(self.shared_buffer_key)
        except posix_ipc.ExistentialError:
            pass

    @coroutine
    def fetch_config_info(self):
        """
        @brief   Retrieve configuration information from the delay configuration server
        """
        log.info(
            "Fetching configuration information from the delay configuration server")
        yield self._delay_client.until_synced()
        sensors = self._delay_client.sensor
        antennas_json = yield sensors.antennas.get_value()
        try:
            antennas = json.loads(antennas_json)
        except Exception as error:
            log.exception("Failed to parse antennas")
            raise error
        self._antennas = [Antenna(antennas[antenna])
                          for antenna in self._ordered_antennas]
        log.info("Ordered the antenna capture list to:\n {}".format(
            "\n".join([i.format_katcp() for i in self._antennas])))
        reference_antenna = yield sensors.reference_antenna.get_value()
        self._reference_antenna = Antenna(reference_antenna)
        log.info("Reference antenna: {}".format(
            self._reference_antenna.format_katcp()))

    @coroutine
    def start(self):
        """
        @brief   Start the delay buffer controller

        @detail  This method will create all necessary posix shared memory segments
                 and semaphores, retreive necessary information from the delay
                 configuration server and start the delay update callback loop.
        """
        log.info("Starting delay buffer controller")
        log.info("Creating IPC buffers and semaphores for delay buffer")
        yield self.fetch_config_info()
        yield self.register_callbacks()
        self.unlink_all()
        # This semaphore is required to protect access to the shared_buffer
        # so that it is not read and written simultaneously
        # The value is set to two such that two processes can read
        # simultaneously
        log.debug("Creating mutex semaphore, key='{}'".format(
            self.mutex_semaphore_key))
        self._mutex_semaphore = posix_ipc.Semaphore(
            self.mutex_semaphore_key,
            flags=posix_ipc.O_CREX,
            initial_value=self._nreaders)

        # This semaphore is used to notify beamformer instances of a change to the
        # delay models. Upon any change its value is simply incremented by one.
        # Note: There sem_getvalue does not work on Mac OS X so the value of this
        # semaphore cannot be tested on OS X (this is only a problem for local
        # testing).
        log.debug("Creating counting semaphore, key='{}'".format(
            self.counting_semaphore_key))
        self._counting_semaphore = posix_ipc.Semaphore(
            self.counting_semaphore_key,
            flags=posix_ipc.O_CREX,
            initial_value=0)

        # This is the share memory buffer that contains the delay models for
        # the
        log.debug("Creating shared memory, key='{}'".format(
            self.shared_buffer_key))
        self._shared_buffer = posix_ipc.SharedMemory(
            self.shared_buffer_key,
            flags=posix_ipc.O_CREX,
            size=C.sizeof(self._delay_model))

        # For reference one can access this memory from another python process using:
        # shm = posix_ipc.SharedMemory("delay_buffer")
        # data_map = mmap.mmap(shm.fd, shm.size)
        # data = np.frombuffer(data_map, dtype=[("delay_rate","float32"),("delay_offset","float32")])
        # data = data.reshape(nbeams, nantennas)
        self._shared_buffer_mmap = mmap(
            self._shared_buffer.fd, self._shared_buffer.size)

        if not self._offline:
            log.info(("Starting delay calculation cycle, "
                      "update rate = {} seconds").format(
                self._update_rate))
            self._update_callback = PeriodicCallback(
                self._safe_update_delays, self._update_rate * 1000)
            self._update_callback.start()
        else:
            self._offline_control_thread = OfflineControlThread(self)
            self._offline_control_thread.start()
            self._offline_control_thread.wait_until_ready()
        log.info("Delay buffer controller started")

    def stop(self):
        """
        @brief   Stop the delay buffer controller

        @detail  This method will stop the delay update callback loop, deregister
                 any sensor callbacks and trigger the closing and unlinking of
                 posix IPC objects.
        """
        log.info("Stopping delay buffer controller")
        if not self._offline:
            self._update_callback.stop()
        else:
            self._offline_control_thread.stop()
            self._offline_control_thread.join()
        self.deregister_callbacks()
        log.debug("Closing shared memory mmap and file descriptor")
        self._shared_buffer_mmap.close()
        self._shared_buffer.close_fd()
        self.unlink_all()
        log.info("Delay buffer controller stopped")

    def _update_phase_reference(self, rt, t, status, value):
        if status != "nominal":
            return
        log.info("Received update to phase-reference: {}, {}, {}, {}".format(
            rt, t, status, value))
        self._phase_reference = Target(value)

    @coroutine
    def register_callbacks(self):
        """
        @brief   Register callbacks on the phase-reference and target positions for each beam

        @detail  The delay configuration server provides information about antennas, reference
                 antennas, phase centres and beam targets. It is currently assumed that the
                 antennas and reference antenna will not (can not) change during an observation
                 as such we here only register callbacks on the phase-reference (a KATPOINT target
                 string specifying the bore sight pointing position) and the individial beam targets.
        """
        log.debug("Registering phase-reference update callback")
        yield self._delay_client.until_synced()
        self._delay_client.sensor.phase_reference.set_sampling_strategy(
            'event')
        self._delay_client.sensor.phase_reference.register_listener(
            self._update_phase_reference)
        for beam in self._ordered_beams:
            sensor_name = "{}_target".format(beam)

            def callback(rt, t, status, value, beam):
                log.debug("Received target update for beam {}: {}".format(
                    beam, value))
                if status == 'nominal':
                    try:
                        self._targets[beam] = Target(value)
                    except Exception as error:
                        log.exception(
                            "Error updating target for beam {}:{}".format(
                                beam, str(error)))
            yield self._delay_client.sensor[
                sensor_name].set_sampling_strategy('event')
            self._delay_client.sensor[sensor_name].register_listener(
                lambda rt, r, status, value, beam=beam: callback(
                    rt, r, status, value, beam))
            self._beam_callbacks[beam] = callback

    def deregister_callbacks(self):
        """
        @brief    Deregister any callbacks started with the register callbacks method
        """
        log.debug("Deregistering phase-reference update callback")
        self._delay_client.sensor.phase_reference.set_sampling_strategy('none')
        self._delay_client.sensor.phase_reference.unregister_listener(
            self._update_phase_reference)
        log.debug("Deregistering targets update callbacks")
        for beam in self._ordered_beams:
            sensor_name = "{}_target".format(beam)
            self._delay_client.sensor[
                sensor_name].set_sampling_strategy('none')
            self._delay_client.sensor[sensor_name].unregister_listener(
                self._beam_callbacks[beam])
        self._beam_callbacks = {}

    def _safe_update_delays(self):
        # This is just a wrapper around update delays that
        # stops it throwing an exception
        try:
            self.update_delays()
        except Exception:
            log.exception("Failure while updating delays")

    def write_delay_model(self, epoch, duration, delays):
        self._delay_model.epoch = epoch
        self._delay_model.duration = duration
        self._delay_model.delays[:] = np.array(delays).astype('float32').ravel()
        self._shared_buffer_mmap.seek(0)
        self._shared_buffer_mmap.write(C.string_at(
            C.addressof(self._delay_model), C.sizeof(self._delay_model)))

    def update_delays(self, epoch=None):
        """
        @brief    Calculate updated delays based on the currently set targets and
                  phase reference.

        @detail   The delays will be calculated in the order specified in the constructor
                  of the class and the delays will be written to the shared memory segment.

                  Two semaphores are used here:
                    - mutex: This is required to stop clients reading the shared
                             memory segment while it is being written to.
                    - counting: This semaphore is incremented after a succesful write
                                to inform clients that there is fresh data to read
                                from the shared memory segment. It is the responsibility
                                of client applications to track the value of this semaphore.
        """
        timer = Timer()
        delay_calc = DelayPolynomial(
            self._antennas, self._phase_reference, self._targets.values(),
            self._reference_antenna)
        if epoch is None:
            epoch = time.time()
        self._delay_model.epoch = epoch
        self._delay_model.duration = self._delay_span
        poly = delay_calc.get_delay_polynomials(
            self._delay_model.epoch, duration=self._delay_span)
        self._delay_model.delays[:] = poly.astype('float32').ravel()
        poly_calc_time = timer.elapsed()
        log.debug("Poly calculation took {} seconds".format(poly_calc_time))
        if poly_calc_time >= self._update_rate:
            log.warning("The time required for polynomial calculation >= delay"
                        " update rate, this may result in degredation of "
                        "beamforming quality")
        timer.reset()
        # Acquire the semaphore for each possible reader
        log.debug("Acquiring semaphore for each reader")
        for ii in range(self._nreaders):
            self._mutex_semaphore.acquire()
        self._shared_buffer_mmap.seek(0)
        self._shared_buffer_mmap.write(C.string_at(
            C.addressof(self._delay_model), C.sizeof(self._delay_model)))
        # Increment the counting semaphore to notify the readers
        # that a new model is available
        log.debug("Incrementing counting semaphore")
        self._counting_semaphore.release()
        # Release the semaphore for each reader
        log.debug("Releasing semaphore for each reader")
        for ii in range(self._nreaders):
            self._mutex_semaphore.release()
        log.debug("Delay model writing took {} seconds on worker side".format(
            timer.elapsed()))
