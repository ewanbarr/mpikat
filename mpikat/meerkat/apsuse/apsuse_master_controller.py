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
import coloredlogs
import json
import tornado
import signal
from subprocess import check_output
from optparse import OptionParser
from tornado.gen import Return, coroutine
from tornado.ioloop import PeriodicCallback
from katcp import AsyncReply, Sensor
from katcp.kattypes import request, return_reply, Str, Float
from katpoint import Target
from mpikat.core.master_controller import MasterController
from mpikat.core.exceptions import ProductLookupError
from mpikat.meerkat.katportalclient_wrapper import KatportalClientWrapper
from mpikat.meerkat.test.utils import MockKatportalClientWrapper
from mpikat.meerkat.apsuse.apsuse_product_controller import ApsProductController
from mpikat.meerkat.apsuse.apsuse_worker_wrapper import ApsWorkerPool

# ?halt message means shutdown everything and power off all machines

log = logging.getLogger("mpikat.apsuse_master_controller")


class ApsMasterController(MasterController):
    """This is the main KATCP interface for the APSUSE
    pulsar searching system on MeerKAT. This controller only
    holds responsibility for capture of data from the CBF
    network and writing of that data to disk.

    This interface satisfies the following ICDs:
    CAM-APSUSE: <link>
    """
    VERSION_INFO = ("mpikat-aps-api", 0, 1)
    BUILD_INFO = ("mpikat-aps-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]

    def __init__(self, ip, port, dummy=False):
        """
        @brief       Construct new ApsMasterController instance

        @params  ip       The IP address on which the server should listen
        @params  port     The port that the server should bind to
        """
        super(ApsMasterController, self).__init__(ip, port, ApsWorkerPool())
        self._katportal_wrapper_type = KatportalClientWrapper
        #self._katportal_wrapper_type = MockKatportalClientWrapper
        self._dummy = dummy
        if self._dummy:
            for ii in range(8):
                self._server_pool.add("127.0.0.1", 50000+ii)

    def setup_sensors(self):
        super(ApsMasterController, self).setup_sensors()
        self._disk_fill_level_sensor = Sensor.float(
            "beegfs-fill-level",
            description="The percentage fill level of the BeeGFS cluster",
            default=0.0,
            unit="percentage",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._disk_fill_level_sensor)

        def check_disk_fill_level():
            try:
                used, avail = map(float, check_output(["df", "/DATA/"]
                    ).splitlines()[1].split()[2:4])
                percent_used = 100.0 * used / (used + avail)
                self._disk_fill_level_sensor.set_value(percent_used)
            except Exception as error:
                log.warning("Failed to check disk usage level: {}".format(
                    str(error)))
        check_disk_fill_level()
        self._disk_fill_callback = PeriodicCallback(
            check_disk_fill_level, 60 * 1000)
        self._disk_fill_callback.start()

    @request(Str(), Str(), Str())
    @return_reply()
    def request_configure(self, req, product_id, streams_json, proxy_name):
        """
        @brief      Configure APSUSE to receive and process data from a subarray

        @detail     REQUEST ?configure product_id antennas_csv n_channels streams_json proxy_name
                    Configure APSUSE for the particular data products

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, which is a useful tag to include
                                      in the data, but should not be analysed further. For example "array_1_bc856M4k".

        @param      streams_json      a JSON struct containing config keys and values describing the streams.

                                      For example:

                                      @code
                                         {'stream_type1': {
                                             'stream_name1': 'stream_address1',
                                             'stream_name2': 'stream_address2',
                                             ...},
                                             'stream_type2': {
                                             'stream_name1': 'stream_address1',
                                             'stream_name2': 'stream_address2',
                                             ...},
                                          ...}
                                      @endcode

                                      The steam type keys indicate the source of the data and the type, e.g. cam.http.
                                      stream_address will be a URI.  For SPEAD streams, the format will be spead://<ip>[+<count>]:<port>,
                                      representing SPEAD stream multicast groups. When a single logical stream requires too much bandwidth
                                      to accommodate as a single multicast group, the count parameter indicates the number of additional
                                      consecutively numbered multicast group ip addresses, and sharing the same UDP port number.
                                      stream_name is the name used to identify the stream in CAM.
                                      A Python example is shown below, for five streams:
                                      One CAM stream, with type cam.http.  The camdata stream provides the connection string for katportalclient
                                      (for the subarray that this APSUSE instance is being configured on).
                                      One F-engine stream, with type:  cbf.antenna_channelised_voltage.
                                      One X-engine stream, with type:  cbf.baseline_correlation_products.
                                      Two beam streams, with type: cbf.tied_array_channelised_voltage.  The stream names ending in x are
                                      horizontally polarised, and those ending in y are vertically polarised.

                                      @code
                                         pprint(streams_dict)
                                         {"cam.http": {"camdata":"http://10.8.67.235/api/client/1"},
                                          "cbf.antenna_channelised_voltage":
                                             {"i0.antenna-channelised-voltage":"spead://239.2.1.150+15:7148"},
                                          ...}
                                      @endcode

                                      If using katportalclient to get information from CAM, then reconnect and re-subscribe to all sensors
                                      of interest at this time.

        @param      proxy_name        The CAM name for the instance of the APSUSE data proxy that is being configured.
                                      For example, "APSUSE_3".  This can be used to query sensors on the correct proxy,
                                      in the event that there are multiple instances in the same subarray.

        @note       A configure call will result in the generation of a new subarray instance in APSUSE that will be added to the clients list.

        @return     katcp reply object [[[ !configure ok | (fail [error description]) ]]]
        """

        msg = ("Configuring new APSUSE product",
               "Product ID: {}".format(product_id),
               "Streams: {}".format(streams_json),
               "Proxy name: {}".format(proxy_name))
        log.info("\n".join(msg))
        # Test if product_id already exists
        if product_id in self._products:
            return ("fail", "APS already has a configured product with ID: {}".format(product_id))

        # Determine number of nodes required based on number of antennas in subarray
        # Note this is a poor way of handling this that may be updated later. In theory
        # there is a throughput measure as a function of bandwidth, polarisations and number
        # of antennas that allows one to determine the number of nodes to run. Currently we
        # just assume one antennas worth of data per NIC on our servers, so two antennas per
        # node.

        streams = json.loads(streams_json)
        try:
            streams['cam.http']['camdata']
        except KeyError as error:
            return ("fail",
                "JSON streams object does not contain required key: {}".format(
                    str(error)))
        @coroutine
        def configure():
            katportal_client = self._katportal_wrapper_type(
                streams['cam.http']['camdata'], product_id)
            self._products[product_id] = ApsProductController(
                self, product_id, katportal_client, proxy_name)
            try:
                yield self._products[product_id].configure()
            except Exception:
                log.exception("Error during configuration")
            self._update_products_sensor()
            log.info("Configured APSUSE instance with ID: {}".format(product_id))
            req.reply("ok",)
        self.ioloop.add_callback(configure)
        raise AsyncReply

    @request(Str())
    @return_reply()
    def request_deconfigure(self, req, product_id):
        """
        @brief      Deconfigure the APSUSE instance.

        @note       Deconfigure the APSUSE instance. If APSUSE uses katportalclient to get information
                    from CAM, then it should disconnect at this time.

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being deconfigured.
                                      For example "array_1_bc856M4k".

        @return     katcp reply object [[[ !deconfigure ok | (fail [error description]) ]]]
        """
        log.info("Deconfiguring APSUSE instace with ID '{}'".format(product_id))
        # Test if product exists
        try:
            product = self._get_product(product_id)
        except ProductLookupError as error:
            return ("fail", str(error))
        try:
            product.deconfigure()
        except Exception as error:
            return ("fail", str(error))
        del self._products[product_id]
        self._update_products_sensor()
        log.info("Deconfigured APSUSE instance with ID '{}'".format(product_id))
        return ("ok",)

    @request(Str(), Str())
    @return_reply()
    @coroutine
    def request_target_start(self, req, product_id, target):
        """
        @brief      Notify APSUSE that a new target is being observed

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".

        @param      target          A KATPOINT target string

        @return     katcp reply object [[[ !target-start ok | (fail [error description]) ]]]
        """
        log.info("Received new target: {}".format(target))
        try:
            product = self._get_product(product_id)
        except ProductLookupError as error:
            raise Return(("fail", str(error)))
        try:
            target = Target(target)
        except Exception as error:
            raise Return(("fail", str(error)))
        yield product.disable_all_writers()
        yield product.enable_writers()
        raise Return(("ok",))

    @request(Str())
    @return_reply()
    def request_capture_start(self, req, product_id):
        """
        @brief      Request that APSUSE start beams streaming

        @detail     Upon this call the provided coherent and incoherent beam configurations will be evaluated
                    to determine if they are physical and can be met with the existing hardware. If the configurations
                    are acceptable then servers allocated to this instance will be triggered to begin production of beams.

        @param      req               A katcp request object

        @param      product_id        This is a name for the data product, used to track which subarray is being deconfigured.
                                      For example "array_1_bc856M4k".

        @return     katcp reply object [[[ !start-beams ok | (fail [error description]) ]]]
        """
        log.info("Capture start requested on product '{}'".format(product_id))
        try:
            product = self._get_product(product_id)
        except ProductLookupError as error:
            return ("fail", str(error))
        @coroutine
        def start():
            try:
                yield product.capture_start()
            except Exception as error:
                log.exception("Error on capture start")
                req.reply("fail", str(error))
            else:
                log.info("Capture start complete for '{}'".format(product_id))
                req.reply("ok",)
        self.ioloop.add_callback(start)
        raise AsyncReply

    @request(Str())
    @return_reply()
    def request_capture_stop(self, req, product_id):
        """
        @brief      Stop APSUSE streaming

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".
        """
        log.info("Capture stop request on '{}'".format(product_id))
        try:
            product = self._get_product(product_id)
        except ProductLookupError as error:
            return ("fail", str(error))

        @coroutine
        def stop():
            yield product.capture_stop()
            log.info("Capture stop complete for '{}'".format(product_id))
            req.reply("ok",)
        self.ioloop.add_callback(stop)
        raise AsyncReply

    @request(Str(), Float())
    @return_reply()
    def request_set_data_rate_per_worker(self, req, product_id, rate):
        """
        @brief      Set the maximum ingest rate per APSUSE worker server

        @detail     This number caps the maximum number of beams that can be
                    ingested into an APSCN node. It is recommended to keep this
                    below 25 Gb/s.

        @param      product_id      This is a name for the data product, used to track which subarray is being deconfigured.
                                    For example "array_1_bc856M4k".

        @param      rate            The data rate per APSCN worker in units of bits/s
        """
        log.info("Set data rate per worker request for product '{}'".format(product_id))
        try:
            product = self._get_product(product_id)
        except ProductLookupError as error:
            return ("fail", str(error))
        try:
            product.set_data_rate_per_worker(rate)
        except Exception as error:
            log.exception("Error when setting data rate per worker: {}".format(str(error)))
            return ("fail", str(error))
        else:
            log.info("Set data rate per worker to {} bits/s".format(rate))
            return ("ok",)

    @request()
    @return_reply()
    def request_register_default_worker_servers(self, req):
        """
        @brief      Add default APSUSE nodes to the server pool
        """
        for idx in range(8):
            self._server_pool.add("apscn{:02d}.mpifr-be.mkat.karoo.kat.ac.za".format(idx), 6000)
        return ("ok",)

@coroutine
def on_shutdown(ioloop, server):
    log.info("Shutting down server")
    yield server.stop()
    ioloop.stop()


def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option('-H', '--host', dest='host', type=str,
        help='Host interface to bind to')
    parser.add_option('-p', '--port', dest='port', type=int,
        help='Port number to bind to')
    parser.add_option('', '--log_level',dest='log_level',type=str,
        help='Port number of status server instance',default="INFO")
    parser.add_option('', '--dummy',action="store_true", dest='dummy',
        help='Set status server to dummy')
    (opts, args) = parser.parse_args()
    logger = logging.getLogger('mpikat')
    coloredlogs.install(
        fmt="[ %(levelname)s - %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
        level=opts.log_level.upper(),
        logger=logger)
    logging.getLogger('katcp').setLevel('INFO')
    ioloop = tornado.ioloop.IOLoop.current()
    log.info("Starting ApsMasterController instance")
    server = ApsMasterController(opts.host, opts.port, dummy=opts.dummy)
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, ioloop, server))

    def start_and_display():
        server.start()
        log.info("Listening at {0}, Ctrl-C to terminate server".format(
            server.bind_address))

    ioloop.add_callback(start_and_display)
    ioloop.start()

if __name__ == "__main__":
    main()

