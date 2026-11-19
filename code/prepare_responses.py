"""Keep historical velocity response epochs; preserve full response stages."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
from lxml import etree as E
HERE=Path(__file__).resolve().parents[1]
source=Path('/home/msseo/works/5.P_wave_moment_tensor_inversion/Response/KS_KG_metadata_1.0.2.xml/KS_KG_metadata_1.0.2.xml')
out=HERE/'inputs/master_velocity.xml'
tree=E.parse(str(source));root=tree.getroot();ns={'s':root.nsmap[None]}
removed=0;kept=0
for net in root.findall('s:Network',ns):
    for sta in net.findall('s:Station',ns):
        for ch in list(sta.findall('s:Channel',ns)):
            if ch.attrib.get('code','') not in ['HHZ','HHN','HHE','ELZ','ELN','ELE']:
                sta.remove(ch);removed+=1
            else:kept+=1
        if not sta.findall('s:Channel',ns):net.remove(sta)
tree.write(str(out),encoding='UTF-8',xml_declaration=True)
merged=Path('/home/msseo/works/27.Haenam_sequence_renewal/docs/decisions/response_inventory_20261014/merged_inventory.xml')
necis=HERE.parent/'sangju_20261117/results/inputs/necis_KS_sangju.xml'
meta=dict(master_source=str(source),master_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 master_creation_date='2024-06-26',velocity_subset=str(out),subset_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
 retained_channel_epochs=kept,removed_other_channel_epochs=removed,
 merged_override=str(merged),merged_sha256=hashlib.sha256(merged.read_bytes()).hexdigest(),
 policy='Exact channel and epoch; audited merged response first, dated NECIS current response second, historical master fallback; require station coordinate agreement within 2 km. HG permitted only with acceleration units. One record per site.',
 limitations='An open historical epoch is not independent proof that the instrument stayed unchanged. The national response history has not received the full Haenam station audit.',
 necis_current=str(necis),necis_current_sha256=hashlib.sha256(necis.read_bytes()).hexdigest())
(HERE/'inputs/response_sources.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(dict(retained=kept,subset_bytes=out.stat().st_size,master_bytes=source.stat().st_size),indent=2))
