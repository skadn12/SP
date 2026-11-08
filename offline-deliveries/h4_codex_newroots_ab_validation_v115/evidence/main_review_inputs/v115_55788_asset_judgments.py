import sys
sys.path.insert(0, '/tmp')
from v115_public_review import save_assets

v = {
 'S-aaa8fe1a3b090c4f': ('contradicts','R02:A:S03:T007',['R02:A:S03:T007'],'Liam explicitly moved rather than deleted the entries, retaining both contributions.'),
 'S-689c505f0c4447df': ('contradicts','R02:A:S01:T007',['R02:A:S01:T007'],'This completed salad uses lemon and olive oil and explicitly leaves balsamic out. Earlier pasta balsamic is a different dish.'),
 'S-c37f7ad3a8609e2a': ('contradicts','R02:A:S02:T005',['R02:A:S02:T005','R02:A:S02:T006'],'Elena instructs a five-minute rest after smoothing; Liam starts a five-minute timer. No fifteen-minute rest is instructed.'),
 'S-cee2650ec69ff094': ('contradicts','R02:B:S03:T009',['R02:B:S03:T009'],'Elena reports the adviser explicitly booked seven tickets and supplied a reference, contradicting a claim the adviser said they could not be booked.'),
 'S-39798672a475faa9': ('supports','R02:A:S02:T005',['R02:A:S02:T005','R02:A:S02:T006'],'Elena explicitly says to whisk smooth and rest five minutes; Liam reports smoothing then starting a five-minute timer.'),
 'S-24ec3e584ebdbbf2': ('supports','R02:B:S01:T007',['R02:B:S01:T003','R02:B:S01:T007','R02:B:S01:T011'],'Loch Fyne is the named seafood restaurant, and the table request is Monday five fifteen in the afternoon, restated17:15 in the note. Primary is the request, not an inferred meal completion.'),
 'S-f616bd526a872d2d': ('contradicts','R02:B:S01:T007',['R02:B:S01:T007','R02:B:S01:T011'],'Requested five fifteen is explicitly afternoon and written17:15;05:15 would be morning and conflicts with the stated request.'),
 'S-ac4abba7e9c9c0e9': ('supports','R02:B:S03:T009',['R02:B:S03:T009'],'Elena explicitly relays the adviser saying seven tickets were booked. This does not assert payment or completed travel.'),
 'S-b5cbf60f3ebe0ba4': ('supports','R02:A:S01:T007',['R02:A:S01:T007'],'Liam explicitly reports finished salad dressing with lemon and olive oil, excluding balsamic.'),
 'S-4d2eb4d5e4234ad0': ('supports','R02:B:S02:T003',['R02:B:S02:T003'],'Elena attributes the east-of-city description to the adviser, without turning it into independent external certification.'),
 'S-b093271bd761cb38': ('contradicts','R02:B:S02:T003',['R02:B:S02:T003'],'The attributed location is explicitly east, contradicting west for the same gallery and enquiry.'),
 'S-77489cb2463adee1': ('supports','R02:A:S03:T007',['R02:A:S03:T007'],'Liam reports moving entries rather than deleting, keeping both contributions. No new playlist creation is claimed.')
}
o = [
 {
  'I1': ('R02:B:S03:T009',['R02:B:S03:T011'],'Adviser booking reference is explicitly relayed atT009 and repeated in the saved-note report atT011. Both reasonable matches are in the final scene after every other item, preserving the same strict order.'),
  'I2': ('R02:A:S03:T005',[],'Liam reports making the named heading here; later location/playback descriptions do not report creating it again.'),
  'I3': ('R02:A:S01:T013',[],'The notes entry is completed here;T011 proposes larger pieces without adding a note yet.'),
  'I4': ('R02:B:S02:T003',[],'Only this turn attributes both museum classification and east location to the adviser.'),
  'I5': ('R02:A:S02:T006',[],'Only this turn reports starting the five-minute timer. Later timer ringing is a different event.'),
  'I6': ('R02:B:S01:T005',[],'Only this turn reports the adviser confirming the restaurant is expensive; the preceding listener question is not that report.')
 },
 {
  'I1': ('R02:A:S02:T012',[],'Liam explicitly reports tasting the first pancake here; invitation to plate and remaining-batch cooking are different states.'),
  'I2': ('R02:B:S01:T011',[],'Only this turn reports entering the reference into the calendar note. Receiving a referenceT009 is a distinct act.'),
  'I3': ('R02:B:S03:T013',[],'Elena explicitly states no fare was quoted here; the previous question supplies no answer.'),
  'I4': ('R02:A:S01:T007',[],'This completed salad-dressing report is unique; the earlier lemon spoonful test is not the bowl-dressing event.'),
  'I5': ('R02:B:S02:T008',[],'Liam reports leaving the date field empty in this note here; Elena not choosing a date is a different actor/action.'),
  'I6': ('R02:A:S03:T007',[],'Only this turn reports moving entries rather than deleting them; preceding suggestion is not a completed move.')
 }
]
save_assets('55788',v,o)
