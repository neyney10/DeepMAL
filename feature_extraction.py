
from feature_extraction.n_bytes_per_packet import NBytesPerPacket
from nfstream import NFStreamer  # https://www.nfstream.org/docs/api


input_pcap_filepath = 'data/DoH-Firefox84-Cloudflare-1.pcap'
plugins = [
    NBytesPerPacket(),
]

my_streamer = NFStreamer(source=input_pcap_filepath,
                                decode_tunnels=True,
                                bpf_filter="udp or tcp",
                                promiscuous_mode=True,
                                snapshot_length=1536,
                                idle_timeout=9999999999,
                                active_timeout=9999999999,
                                accounting_mode=3,
                                udps=plugins,
                                n_dissections=20,
                                statistical_analysis=True,
                                splt_analysis=0,
                                n_meters=0,
                                performance_report=0)

my_streamer.to_csv('output.csv')
print('Exiting...')