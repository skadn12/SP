import sys
sys.path.insert(0, '/tmp')
from v115_public_review import save_assets

v = {
 'S-4509de1fd9620fad': ('contradicts','R02:B:S02:T009',['R02:B:S02:T007','R02:B:S02:T009'],'Emily explicitly says the adviser did not say admission was free and could not supply a charge. Unknown fee is not a free-admission assertion.'),
 'S-5731ad1dee86e8dd': ('supports','R02:A:S01:T005',['R02:A:S01:T004','R02:A:S01:T005'],'Emily assigns the shallow bottom-shelf tray to cars, and Liam then reports putting the cars in that tray. Primary is the completed-action report; preceding instruction resolves the tray.'),
 'S-a65dac0f3fdab2a8': ('contradicts','R02:B:S03:T009',['R02:B:S03:T009'],'Emily crosses out21:09 and writes20:09 as the corrected final departure, explicitly superseding the first time.'),
 'S-020f5af571ad4c5f': ('supports','R02:A:S02:T008',['R02:A:S02:T006','R02:A:S02:T007','R02:A:S02:T008'],'Liam has dried the plastic items, Emily identifies the low kitchen cupboard, and Liam reports they are in that cupboard. Primary records completion; earlier turns resolve object and location.'),
 'S-3dc99e4dec533c27': ('contradicts','R02:A:S02:T007',['R02:A:S02:T006','R02:A:S02:T007','R02:A:S02:T008'],'Beside the red toy box was a question, explicitly rejected. Emily assigns the kitchen cupboard and Liam reports putting the items there.'),
 'S-fe2499ccacefa265': ('supports','R02:B:S03:T009',['R02:B:S03:T009'],'Emily explicitly keeps20:09 withTR2602 as corrected departure after crossing out the superseded21:09.'),
 'S-34aee86aa1ab947d': ('contradicts','R02:B:S01:T007',['R02:B:S01:T007'],'The table request is Monday eleven thirty in the morning, not at night. This judges the request rather than claiming independent confirmation of every parameter.'),
 'S-83ed06d32d74c4de': ('supports','R02:B:S02:T009',['R02:B:S02:T007','R02:B:S02:T009'],'Emily relays unavailable entrance-fee information and explicitly says the adviser quoted no price. The proposition is about what was supplied, not the actual unknown fee.'),
 'S-1495989dca86de7f': ('contradicts','R02:A:S03:T005',['R02:A:S03:T004','R02:A:S03:T005'],'Passing is expressly allowed and written on the rule card, contradicting a requirement to answer without passing.'),
 'S-43b8541214360027': ('supports','R02:A:S03:T005',['R02:A:S03:T005'],'Liam reports writing Passing is allowed on the rule card; no absolute duty to answer is introduced.'),
 'S-4d3ebe11da069e61': ('contradicts','R02:A:S01:T004',['R02:A:S01:T004','R02:A:S01:T005'],'Red box is reserved for bricks; cars are assigned to the bottom-shelf tray and Liam reports using it.'),
 'S-f6a4fad3641f0fee': ('supports','R02:B:S01:T007',['R02:B:S01:T007'],'Emily explicitly requests Monday11:30 in the morning. A request is not silently upgraded to independently confirmed booking parameters.')
}
o = [
 {
  'I1': ('R02:A:S03:T005', [], 'Only this turn reports Liam writing the passing rule; later desire to keep it is not a second writing event.'),
  'I2': ('R02:B:S03:T009', [], 'Only this turn reports crossing out the superseded21:09, after the adviser correction.'),
  'I3': ('R02:A:S01:T007', [], 'Only this turn reports the tablet photograph of the tower; the prior question proposes a photo without completing one.'),
  'I4': ('R02:B:S02:T007', [], 'Exact telephone digits appear in this attributed adviser report; later references to saving the number do not relay the digits again.'),
  'I5': ('R02:A:S02:T010', [], 'Liam reports completed wiping including the sticky bit; the previous suggestion is not completion.'),
  'I6': ('R02:B:S01:T005', [], 'Emily supplies the attributed Napier Street City Centre location only here.')
 },
 {
  'I1': ('R02:B:S01:T011', [], 'Only Emily turn explicitly relaying the exact restaurant reference; prior question has no number.'),
  'I2': ('R02:A:S02:T012', [], 'Liam reports the tick completed here; Emily permission to tick is not the same action report.'),
  'I3': ('R02:B:S03:T011', [], 'Only this turn explicitly says the booking offer was declined; information-only intent does not itself report declining.'),
  'I4': ('R02:A:S01:T012', [], 'Emily explicitly explains start signal versus finish deadline here. Liam earlier describes starting but is a different actor and speech act.'),
  'I5': ('R02:B:S02:T011', [], 'Only this turn reports Emily saving the phone number; giving digits and planning to ask the pool are different speech acts.'),
  'I6': ('R02:A:S03:T013', [], 'Liam reports the two unused cards placed in an envelope here; preceding statement merely says two remain waiting.')
 }
]
save_assets('45849', v, o)
