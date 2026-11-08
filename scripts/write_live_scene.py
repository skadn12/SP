import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/autodl-tmp/gxy-selfplay')
B=ROOT/'research-protocols/selfplay_week_20261019/postweek_live_extension_v4'
HAND=B/'handoff'
def bind(p):
 p=Path(p).resolve();d=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(d).hexdigest(),'bytes':len(d)}
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def scene(root,route,index,author,texts,native=None):
  sref=f'LEX1:{route}:S{index:02d}'; src=ROOT/f'offline-deliveries/h4_codex_train_pool_v125/sources/{root}.json'
  hand=HAND/root/'SOURCE_AUTHOR_INPUT.json'; prior=src
  # caller may provide prior cumulative source for later scenes
  if route=='A' and index>1: prior=B/'roots'/root/f'LEX1_A_S{index-1:02d}'/'CUMULATIVE_SOURCE.json'
  if route=='B': prior=B/'roots'/root/f'LEX1_A_S03'/'CUMULATIVE_SOURCE.json' if index==1 else B/'roots'/root/f'LEX1_B_S{index-1:02d}'/'CUMULATIVE_SOURCE.json'
  turns=[{'turn_ref':f'{sref}:T{i:03d}','scene_ref':sref,'session_ref':sref,'speaker':sp,'text':tx} for i,(sp,tx) in enumerate(texts,1)]
  draft={'schema_version':'live_extension_scene_draft_v1','root_id':root,'scene_ref':sref,'author_id':author,'created_at_utc':now(),
   'input_source_binding':bind(prior),'author_input_binding':bind(hand),'turns':turns}
  if native:
   draft['native_event_binding']=bind(B/'inputs'/f'{root}_B_S{index:02d}_EVENT.json');draft['native_event_source_id']=native
  folder=B/'roots'/root/sref.replace(':','_');folder.mkdir(parents=True,exist_ok=True)
  dp=folder/'DRAFT.json';save(dp,draft)
  base=json.load(open(prior)); cum={'schema_version':'live_extension_cumulative_source_v1','root_id':root,'turns':base['turns']+turns,
   'baseline_binding':bind(src),'input_source_binding':bind(prior),'draft_binding':bind(dp),'scene_ref':sref}
  cp=folder/'CUMULATIVE_SOURCE.json';save(cp,cum)
  save(folder/'AUTHOR_CONSUMPTION.json',{'schema_version':'source_author_consumption_v1','author_id':author,'root_id':root,'scene_ref':sref,'full_source_read':True,'author_input_binding':bind(hand),'input_source_binding':bind(prior),'material_targets_read':True,'solver_outputs_read':False,'created_at_utc':now(),'draft_binding':bind(dp)})
  print(json.dumps({'root':root,'scene':sref,'draft':bind(dp),'source':bind(cp)},ensure_ascii=False))

scenes={
 'cc-train-episode-20432': [('Liam','Mom, let us fix the times before we put our plans on the calendar.'),('Emily','Good idea. The river afternoon is on this Saturday, and I wrote two o\'clock, 14:00, for our meeting by the water.'),('Liam','So 14:00 is the planned start of that outing, not the time a train reaches Ely.'),('Emily','Exactly. The Ely trip is the following Saturday, a separate plan.'),('Liam','What time does that train leave Cambridge?'),('Emily','The rail adviser has now confirmed a 10:42 departure from Cambridge.'),('Liam','And the arrival time we already recorded was 12:07.'),('Emily','Yes, 10:42 is departure and 12:07 is arrival for four people.'),('Liam','That means the train leaves before the river plan, but on the later Saturday.'),('Emily','Right. I have not changed the river appointment or moved it to the train day.'),('Liam','Did the adviser say the ticket was paid for?'),('Emily','No. The booking reference is recorded, but the fare is still payable at the station.'),('Liam','I will write 10:42 departure, 12:07 arrival, and 14:00 river plan in separate lines.'),('Emily','That will keep the times, dates, and payment status from being mixed together.')],
 'cc-train-episode-19708': [('Emily','Marco, let us write the exact clock readings down before we compare your arrangements.'),('Marco','The barbecue arrival is around 14:00, but that is an approximate arrival time, not a departure.'),('Emily','The taxi request has an arrival limit of 13:00, so we must not call that a pickup time.'),('Marco','I have now checked the taxi service and the pickup is 12:20, with arrival by 13:00.'),('Emily','Good. The train enquiry also gave a new exact departure at 18:42 on Friday, rather than only saying after 18:15.'),('Marco','Its arrival is 19:31, and that is the train trip, not the barbecue.'),('Emily','So 12:20 is the taxi departure, 13:00 its arrival limit, and 14:00 is the barbecue arrival.'),('Marco','Then 18:42 is the Friday train departure, with 19:31 as its arrival.'),('Emily','Keep the booking references beside their own service; a quoted fare is not proof of payment.'),('Marco','The taxi is arranged for the outing, while the train reference is confirmed for three people.'),('Emily','And the barbecue time remains approximate, so it should not be presented as an exact appointment.'),('Marco','I will record the four exact or explicitly bounded readings with their event labels.'),('Emily','That gives us a defensible order without turning an arrival deadline into a departure.'),('Marco','I have checked each line against the separate enquiries and will ask if anything is unclear.')]
}
if __name__=='__main__':
 root=sys.argv[1];scene(root,'A',1,'/root/source-author-'+root.split('-')[-1],scenes[root])
