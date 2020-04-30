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
import numpy as np
from mpikat.core.ip_manager import ip_range_from_stream

log = logging.getLogger('mpikat.apsuse_config_manager')

DEFAULT_DATA_RATE_PER_WORKER = 20e9  # bits / s

DUMMY_FBF_CONFIG = {
    "coherent-beam-multicast-groups":"spead://239.11.1.15+15:7147",
    "coherent-beam-multicast-groups-data-rate": 7e9,
    "incoherent-beam-multicast-group": "spead://239.11.1.14:7147",
    "incoherent-beam-multicast-group-data-rate": 150e6,
}


HOST_TO_LEAF_MAP = {
    "apscn00.mpifr-be.mkat.karoo.kat.ac.za": 1,
    "apscn01.mpifr-be.mkat.karoo.kat.ac.za": 1,
    "apscn02.mpifr-be.mkat.karoo.kat.ac.za": 1,
    "apscn03.mpifr-be.mkat.karoo.kat.ac.za": 1,
    "apscn04.mpifr-be.mkat.karoo.kat.ac.za": 0,
    "apscn05.mpifr-be.mkat.karoo.kat.ac.za": 0,
    "apscn06.mpifr-be.mkat.karoo.kat.ac.za": 0,
    "apscn07.mpifr-be.mkat.karoo.kat.ac.za": 0,
}


class ApsConfigurationError(Exception):
    pass


class ApsWorkerBandwidthExceeded(Exception):
    pass


class ApsWorkerTotalBandwidthExceeded(Exception):
    pass


class ApsWorkerConfig(object):
    def __init__(self, total_bandwidth=DEFAULT_DATA_RATE_PER_WORKER):
        log.debug("Created new apsuse worker config")
        self._total_bandwidth = total_bandwidth
        self._available_bandwidth = self._total_bandwidth
        self._incoherent_groups = []
        self._coherent_groups = []
        self._incoherent_beams = []
        self._coherent_beams = []
        self._even = True

    def set_even(self, even_odd):
        self._even = even_odd

    def can_use_host(self, hostname):
        HOST_TO_LEAF_MAP[hostname] = int(self._even)

    def add_incoherent_group(self, group, bandwidth):
        if bandwidth > self._total_bandwidth:
            log.debug("Adding group would exceed worker bandwidth")
            raise ApsWorkerTotalBandwidthExceeded

        if self._available_bandwidth < bandwidth:
            log.debug("Adding group would exceed worker bandwidth")
            raise ApsWorkerBandwidthExceeded
        else:
            log.debug("Adding group {} to worker".format(group))
            self._incoherent_groups.append(group)
            self._available_bandwidth -= bandwidth

    def add_coherent_group(self, group, bandwidth):
        if self._available_bandwidth < bandwidth:
            log.debug("Adding group would exceed worker bandwidth")
            raise ApsWorkerBandwidthExceeded
        else:
            self._coherent_groups.append((group))
            log.debug("Adding group {} to worker".format(group))
            self._available_bandwidth -= bandwidth

    def data_rate(self):
        return self._total_bandwidth - self._available_bandwidth

    def coherent_groups(self):
        return self._coherent_groups

    def incoherent_groups(self):
        return self._incoherent_groups

    def coherent_beams(self):
        return self._coherent_beams

    def incoherent_beams(self):
        return self._incoherent_beams


class ApsConfigGenerator(object):
    def __init__(self, fbfuse_config,
        bandwidth_per_worker=DEFAULT_DATA_RATE_PER_WORKER):
        self._fbfuse_config = fbfuse_config
        self._bandwidth_per_worker = bandwidth_per_worker

        self._incoherent_range = ip_range_from_stream(
            self._fbfuse_config['incoherent-beam-multicast-group'])
        self._incoherent_mcast_group_rate = (
            self._fbfuse_config['incoherent-beam-multicast-group-data-rate'])
        self._incoherent_groups = list(self._incoherent_range)

        self._coherent_range = ip_range_from_stream(
            self._fbfuse_config['coherent-beam-multicast-groups'])
        self._coherent_mcast_group_rate = (
            self._fbfuse_config['coherent-beam-multicast-groups-data-rate'])
        self._coherent_groups = list(self._coherent_range)

        self._even_groups = list(np.array(self._coherent_groups)[::2])
        self._odd_groups = list(np.array(self._coherent_groups)[1::2])

    def allocate_groups(self, server):
        leaf = HOST_TO_LEAF_MAP[server.hostname]
        if leaf == 0:
            cgroups = self._even_groups
        else:
            cgroups = self._odd_groups
        current_worker = ApsWorkerConfig(self._bandwidth_per_worker)
        if self._incoherent_groups:
            group = self._incoherent_groups.pop(0)
            try:
                current_worker.add_incoherent_group(group, self._incoherent_mcast_group_rate)
            except (ApsWorkerTotalBandwidthExceeded, ApsWorkerBandwidthExceeded):
                log.error("Incoherent beam mutlicast group ({} Gb/s) size exceeds data rate for one node ({} Gb/s)".format(
                    self._incoherent_mcast_group_rate/1e9, current_worker._total_bandwidth/1e9))
                log.error("Incoherent beam data will not be captured")

        while cgroups:
            group = cgroups[0]
            try:
                current_worker.add_coherent_group(group, self._coherent_mcast_group_rate)
            except ApsWorkerTotalBandwidthExceeded:
                log.error("Coherent beam mutlicast group ({} Gb/s) size exceeds data rate for one node ({} Gb/s)".format(
                    self._coherent_mcast_group_rate/1e9, current_worker._total_bandwidth/1e9))
                log.error("Coherent beam data will not be captured")
                break
            except ApsWorkerBandwidthExceeded:
                return self._finalise_worker(current_worker, server)
            else:
                cgroups.pop(0)

    def _finalise_worker(self, worker, server):
        valid = False
        for incoherent_group in worker.incoherent_groups():
            valid = True
            worker._incoherent_beams.append("ifbf00000")
        for coherent_group in worker.coherent_groups():
            valid = True
            spead_formatted = "spead://{}:{}".format(str(coherent_group), self._coherent_range.port)
            mapping = json.loads(self._fbfuse_config['coherent-beam-multicast-group-mapping'])
            beam_idxs = mapping[spead_formatted]
            worker._coherent_beams.extend(beam_idxs)
        log.debug(("Worker {} config: coherent-groups: {},"
                   " coherent-beams: {}, incoherent-groups: {},"
                   " incoherent-beams: {},").format(
                   str(server), map(str, worker.coherent_groups()),
                   map(str, worker.coherent_beams()),
                   map(str, worker.incoherent_groups()),
                   map(str, worker.incoherent_beams())))
        if valid:
            return worker
        else:
            return None

    def remaining_incoherent_groups(self):
        return self._incoherent_groups

    def remaining_coherent_groups(self):
        return self._even_groups + self._odd_groups
