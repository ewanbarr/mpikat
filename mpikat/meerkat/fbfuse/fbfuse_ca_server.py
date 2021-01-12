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
import tornado
import signal
import time

from tornado.gen import Return
from optparse import OptionParser
from katcp import Sensor, Message, AsyncDeviceServer
from katcp.kattypes import request, return_reply, Str, Int
from katportalclient import KATPortalClient
from katpoint import Antenna, Target

from sqlalchemy import Column, Text, or_, and_
from sqlalchemy.dialects.mysql import INTEGER, TINYINT
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
Base = declarative_base()
metadata = Base.metadata
DEFAULT_KEY = "any"

log = logging.getLogger("mpikat.fbfuse_ca_server")


class ScheduleBlockConfiguration(Base):
    __tablename__ = 'schedule_block_configuration'

    id = Column(INTEGER(11), primary_key=True)
    mkat_pid = Column(Text)
    sb_id = Column(Text)
    config_json = Column(Text, nullable=False)

    @property
    def config(self):
        return json.loads(self.config_json)


class TargetConfiguration(Base):
    __tablename__ = 'target_configuration'

    id = Column(INTEGER(11), primary_key=True)
    source_name = Column(Text)
    mkat_pid = Column(Text)
    sb_id = Column(Text)
    config_json = Column(Text, nullable=False)

    @property
    def config(self):
        return json.loads(self.config_json)



class BaseFbfConfigurationAuthority(AsyncDeviceServer):
    """This is an example/template for how users
    may develop an fbf configuration authority server
    """
    VERSION_INFO = ("mpikat-fbf-ca-api", 0, 1)
    BUILD_INFO = ("mpikat-fbf-ca-implementation", 0, 1, "rc1")
    DEVICE_STATUSES = ["ok", "degraded", "fail"]
    def __init__(self, ip, port):
        super(BaseFbfConfigurationAuthority, self).__init__(ip, port)
        self._configuration_sensors = {}
        self._configuration_callbacks = {}
        self._products = {}


        self._sb_ids = {}
        self._mkat_pids = {}
        self._kcp_urls = {}

    def start(self):
        """
        @brief  Start the BaseFbfConfigurationAuthority server
        """
        super(BaseFbfConfigurationAuthority,self).start()

    def setup_sensors(self):
        """
        @brief  Set up monitoring sensors.

        @note   The following sensors are made available on top of default sensors
                implemented in AsynDeviceServer and its base classes.

                device-status:  Reports the health status of the CA server and associated devices:
                                Among other things report HW failure, SW failure and observation failure.

        """
        self._device_status = Sensor.discrete(
            "device-status",
            description="Health status of this device",
            params=self.DEVICE_STATUSES,
            default="ok",
            initial_status=Sensor.UNKNOWN)
        self.add_sensor(self._device_status)


    @request(Str(), Str())
    @return_reply()
    def request_set_kpc_url(self, req, product_id, kcp_url):
        """
        Set the KATPortal API endpoint to be used product lookups

        @param      product_id  The product identifier
        @param      kcp_url  KATPortal API endpoint, e.g. http://10.97.1.14/api/client/1
        """
        self._kcp_urls[product_id] = kcp_url
        return ("ok", )

    @request(Str(), Str(), Str(), Int(), Str())
    @return_reply(Str())
    @tornado.gen.coroutine
    def request_get_schedule_block_configuration(self, req, product_id, sb_id, mkat_pid, nchans, kpc_url):
        """
        @brief      Get an FBFUSE configuration for the current instance

        @param      product_id  The product identifier
        @param      sb_id       The schedule block identifier
        @param      mkat_pid    The MeerKAT project ID
        @param      nchans      The CBF F-engine number of channels
        @param      kpc_url     The URL for the current KATPORTAL instance

        @note       The product_id argument may be superfluous, although it allows
                    the CA server to look up parameters on the unconfigured product
                    from the FBFUSE sensor set through katportalclient
        """
        log.info("Received SB configuration request for '{}' with arguments ({})".format(
            product_id, ", ".join(map(str, [sb_id, mkat_pid, nchans, kpc_url]))))
        if product_id in self._configuration_sensors:
            self.remove_sensor(self._configuration_sensors[product_id])
            del self._configuration_sensors[product_id]
            self.mass_inform(Message.inform('interface-changed'))
        self._products[product_id] = {
            "sb_id": sb_id,
            "mkat_pid": mkat_pid,
            "nchans": nchans,
            "kpc_url": kpc_url
        }
        config = yield self.get_sb_config(product_id, sb_id, mkat_pid)
        raise Return(("ok", json.dumps(config)))

    @tornado.gen.coroutine
    def get_sb_config(self, product_id, sb_id, mkat_pid):
        raise NotImplemented

    @request(Str(), Str())
    @return_reply()
    @tornado.gen.coroutine
    def request_target_configuration_start(self, req, product_id, target_string):
        """
        @brief      Set up a beam configuration sensor for the FBFUSE instance

        @param      product_id     The product identifier
        @param      target_string  A KATPOINT target string (boresight pointing position)
        """
        log.info("Received target configuration request for '{}' with target: {}".format(product_id, target_string))
        if not product_id in self._configuration_sensors:
            log.debug("Creating configuration sensor for '{}'".format(product_id))
            self._configuration_sensors[product_id] = Sensor.string(
                "{}-beam-position-configuration".format(product_id),
                description="Configuration description for FBF beam placement",
                default="",
                initial_status=Sensor.UNKNOWN)
            self.add_sensor(self._configuration_sensors[product_id])
            self.mass_inform(Message.inform('interface-changed'))
        initial_config = yield self.get_target_config(product_id, target_string)
        self.update_target_config(product_id, initial_config)
        raise Return(("ok",))

    @tornado.gen.coroutine
    def get_target_config(self, product_id, target):
        # This should call update target config
        raise NotImplemented

    def update_target_config(self, product_id, config):
        log.debug("Updating target config on '{}' with config: {}".format(product_id, config))
        self._configuration_sensors[product_id].set_value(json.dumps(config))


class DefaultConfigurationAuthority(BaseFbfConfigurationAuthority):
    def __init__(self, host, port):
        super(DefaultConfigurationAuthority, self).__init__(host, port)
        self.default_config = {
            u'coherent-beams-nbeams':100,
            u'coherent-beams-tscrunch':16,
            u'coherent-beams-fscrunch':1,
            u'coherent-beams-granularity':6,
            u'incoherent-beam-tscrunch':16,
            u'incoherent-beam-fscrunch':1,
            }

    @tornado.gen.coroutine
    def get_target_config(self, product_id, target):
        # Return just a boresight beam
        raise Return({"beams":[target]})

    @tornado.gen.coroutine
    def get_sb_config(self, product_id, sb_id, mkat_pid):
        raise Return(self.default_config)

    @request(Str())
    @return_reply()
    def request_update_default_sb_config(self, req, config_json):
        """
        @brief      Update the default config returned on a get_sb_config call.
                    This is intended for testing purposes only.
        """
        self.default_config.update(json.loads(config_json))
        return ("ok",)


class TrapumConfigurationAuthority(BaseFbfConfigurationAuthority):
    DEFAULT_ANTENNAS = ",".join("m{:03d}".format(i) for i in range(64))

    def __init__(self, host, port, database):
        super(TrapumConfigurationAuthority, self).__init__(host, port)
        self.default_sb_config = {
            u'coherent-beams-nbeams': 480,
            u'coherent-beams-tscrunch': 16,
            u'coherent-beams-fscrunch': 1,
            u'coherent-beams-antennas': self.DEFAULT_ANTENNAS,
            u'coherent-beams-granularity': 6,
            u'incoherent-beam-tscrunch': 16,
            u'incoherent-beam-fscrunch': 1,
            u'incoherent-beam-antennas': self.DEFAULT_ANTENNAS
            }
        self.default_tg_config = {
            "beams": [],
            "tilings": [
                {
                    "nbeams": 288,
                    "target": None,
                    "overlap": 0.7
                }
            ]
        }
        self._session_engine = create_engine(database,
            echo=False, poolclass=NullPool)
        self._session_factory = sessionmaker(
            bind=self._session_engine)
        log.info("Default SB configuration: {}".format(
            self.default_sb_config))

    @contextmanager
    def session(self):
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as error:
            session.rollback()
            raise error
        finally:
            session.close()

    @tornado.gen.coroutine
    def get_target_config(self, product_id, katpoint_target):
        log.info("Fetching TG configuration")
        config = self.default_tg_config.copy()
        if product_id not in self._products:
            log.error("Received target requested without being configured for a schedule block")
            return config
        mkat_pid = self._products[product_id]["mkat_pid"]
        sb_id = self._products[product_id]["sb_id"]
        target = Target(katpoint_target)
        source = target.name
        check_list = [
            [source,      sb_id,       mkat_pid],
            [source,      sb_id,       DEFAULT_KEY],
            [source,      DEFAULT_KEY, mkat_pid],
            [source,      DEFAULT_KEY, DEFAULT_KEY],
            [DEFAULT_KEY, sb_id,       mkat_pid],
            [DEFAULT_KEY, sb_id,       DEFAULT_KEY],
            [DEFAULT_KEY, DEFAULT_KEY, mkat_pid],
            [DEFAULT_KEY, DEFAULT_KEY, DEFAULT_KEY]
        ]
        config =  self.default_tg_config.copy()
        with self.session() as s:
            for source_match, sb_id_match, mkat_pid_match in check_list:
                log.info("Checking for target config [{}, {}, {}]".format(
                    source_match, sb_id_match, mkat_pid_match))
                result = s.query(TargetConfiguration).filter(
                        TargetConfiguration.source_name.ilike(source_match),
                        TargetConfiguration.sb_id.ilike(sb_id_match),
                        TargetConfiguration.mkat_pid.ilike(mkat_pid_match)
                    ).first()
                if result:
                    log.info("Config found")
                    break
            if result is not None:
                config.update(result.config)
            for tiling in config["tilings"]:
                if not tiling["target"]:
                    tiling["target"] = katpoint_target
        log.info("Final target config: {}".format(config))
        raise Return(config)

    @tornado.gen.coroutine
    def get_sb_config(self, product_id, sb_id, mkat_pid):
        log.info("Fetching SB configuration")
        check_list = [
            [sb_id,       mkat_pid],
            [sb_id,       DEFAULT_KEY],
            [DEFAULT_KEY, mkat_pid],
            [DEFAULT_KEY, DEFAULT_KEY],
        ]
        config = self.default_sb_config.copy()
        with self.session() as s:
            for sb_id_match, mkat_pid_match in check_list:
                log.info("Checking for SB config [{}, {}]".format(
                    sb_id_match, mkat_pid_match))
                result = s.query(ScheduleBlockConfiguration).filter(
                        ScheduleBlockConfiguration.sb_id.ilike(sb_id_match),
                        ScheduleBlockConfiguration.mkat_pid.ilike(mkat_pid_match)
                    ).first()
                if result:
                    log.info("Config found")
                    config.update(result.config)
                    break
            log.info("Final target config: {}".format(config))
        raise Return(config)


@tornado.gen.coroutine
def on_shutdown(ioloop, server):
    log.info("Shutting down server")
    yield server.stop()
    ioloop.stop()

def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option('-H', '--host', dest='host', type=str,
        help='Host interface to bind to')
    parser.add_option('-p', '--port', dest='port', type=long,
        help='Port number to bind to')
    parser.add_option('-d', '--db', dest='database', type=str,
        help='SQLA compatible database definition')
    parser.add_option('', '--log_level',dest='log_level',type=str,
        help='Port number of status server instance',default="INFO")
    (opts, args) = parser.parse_args()
    FORMAT = "[ %(levelname)s - %(asctime)s - %(filename)s:%(lineno)s] %(message)s"
    log.setLevel(opts.log_level.upper())
    for handler in logging.getLogger('').handlers:
        handler.setFormatter(logging.Formatter(FORMAT))
    ioloop = tornado.ioloop.IOLoop.current()
    log.info("Starting DefaultConfigurationAuthority instance")
    server = TrapumConfigurationAuthority(opts.host, opts.port, opts.database)
    signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(
        on_shutdown, ioloop, server))
    def start_and_display():
        server.start()
        log.info("Listening at {0}, Ctrl-C to terminate server".format(server.bind_address))
    ioloop.add_callback(start_and_display)
    ioloop.start()

if __name__ == "__main__":
    main()












