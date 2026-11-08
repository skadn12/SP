# H4 Necessity Canary QA Probe v1

Runs the existing relation-aware QA Writer wrapper and the unchanged local QA
replay on a source-ready H4 canary. Composer is deliberately not called in
this stage; the output is frozen atomic QA plus grounding/K=5 evidence for the
next relation-guided composition stage.
