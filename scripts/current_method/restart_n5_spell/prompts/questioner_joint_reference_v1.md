# Long-Dialogue Joint Question Generation Reference

## Provenance

The examples below are newly authored synthetic examples created specifically for this task.

They are not copied from QuALITY, HotpotQA, MuSiQue, RACE, or any other dataset.

They do not contain facts, entities, events, or target relations from the dialogue that you will analyze.

Their purpose is to demonstrate the difference between:

* a genuine joint question that requires complementary evidence; and
* a superficially difficult question that is actually answerable from one local passage.

Do not copy the wording or story patterns mechanically. Apply the underlying reasoning principles to the target dialogue.

---

## 1. What counts as a genuine joint question?

A genuine joint question must require at least two nonredundant evidence groups.

The evidence groups must contribute different necessary information.

A candidate is valid only when all of the following are true:

1. No single selected turn directly contains the complete answer.
2. Evidence group A alone is insufficient to answer the question uniquely.
3. Evidence group B alone is insufficient to answer the question uniquely.
4. Combining A and B makes the answer uniquely recoverable.
5. A and B are not merely repetitions or paraphrases of the same fact.
6. The relationship between A and B is natural and meaningful.
7. The question does not reveal one of the required reasoning steps in its wording.
8. The selected evidence is the smallest sufficient evidence set.

Long distance between turns is not enough.

A large number of cited turns is not enough.

A difficult-looking question is not enough.

---

## 2. Recommended generation process

For each candidate, follow this process.

### Step 1: Find complementary evidence

Look for two evidence groups that play different roles.

Examples of useful relationships include:

* earlier background and later requirement;
* prior experience and later decision;
* initial goal and later implementation;
* claim and later verification;
* two independent constraints;
* contributions from different people;
* earlier prediction and later outcome;
* comparison across two situations;
* information from one stage that explains a choice in another stage.

Do not begin by choosing a broad motivation statement merely because it is salient.

### Step 2: Construct the target answer

Write the complete answer that becomes available only after combining both evidence groups.

The answer should express the relationship between the evidence, not merely repeat one sentence from the dialogue.

### Step 3: Perform single-side ablation

Test the candidate using only evidence A.

Then test it using only evidence B.

Reject the candidate when either side alone supports the complete answer.

### Step 4: Write the question

Write a natural English reading-comprehension question whose answer is the combined conclusion.

The question should not explicitly state the intermediate conclusion that the reader is expected to infer.

### Step 5: Check for shortcuts

Reject the candidate if:

* one turn already states the answer;
* the question asks only about an early fact;
* later turns merely repeat the same fact;
* the question itself supplies one required inference;
* unrelated facts were joined only to make the question appear complex;
* one evidence group is only background decoration.

---

## 3. Positive examples

### Positive Example 1: Background and later requirement

Evidence A:

Elena explains that she spent two years translating hospital discharge instructions and medical consent forms.

Evidence B:

A community clinic later announces a volunteer position requiring previous experience translating medical documents.

Question:

Which earlier experience makes Elena qualified for the clinic's volunteer position?

Answer:

Her two years of experience translating hospital discharge instructions and medical consent forms.

Why evidence A alone is insufficient:

It establishes Elena's experience but does not identify the later position or its requirement.

Why evidence B alone is insufficient:

It establishes the position's requirement but does not show whether Elena satisfies it.

Why the combination works:

The earlier experience can be matched with the later requirement.

Reasoning type:

background_requirement_matching

---

### Positive Example 2: Earlier result and later decision

Evidence A:

Omar reports that introducing a free weekend shuttle increased attendance among older residents at a health center.

Evidence B:

The town later considers two ways to improve attendance at a new clinic: distributing more flyers or operating a weekend shuttle.

Question:

Which proposed action is better supported by Omar's earlier observation?

Answer:

Operating the weekend shuttle.

Why evidence A alone is insufficient:

It reports a previous result but does not identify the current alternatives.

Why evidence B alone is insufficient:

It lists the alternatives but provides no evidence showing which one is more likely to work.

Why the combination works:

The earlier observed result provides evidence for choosing one of the later alternatives.

Reasoning type:

prior_evidence_later_choice

---

### Positive Example 3: Report and later verification

Evidence A:

Lena reports that the playground lights repeatedly stop working after heavy rain.

Evidence B:

An electrician later finds water entering a damaged junction box, while the maintenance log confirms that the outages occurred after storms.

Question:

Which earlier complaint was later verified, and what evidence explained the problem?

Answer:

Lena's complaint that the playground lights failed after heavy rain was verified by the maintenance log, and water entering the damaged junction box explained the failures.

Why evidence A alone is insufficient:

It contains only an unverified complaint.

Why evidence B alone is insufficient:

It provides later technical and recorded evidence but does not identify the earlier reported concern being verified.

Why the combination works:

The earlier report identifies the claim, while the later evidence verifies and explains it.

Reasoning type:

source_verification_synthesis

---

## 4. Negative examples

### Negative Example 1: Single-turn shortcut

Evidence A:

Marcus says, "I joined the library committee because my daughter could not find a quiet place to study."

Evidence B:

Marcus later attends committee meetings and helps draft a renovation plan.

Bad question:

Why did Marcus join the library committee?

Why this candidate must be rejected:

Evidence A directly states the complete answer. Evidence B is only a later consequence and contributes no necessary information.

Failure type:

single_turn_shortcut

---

### Negative Example 2: Repeated evidence disguised as multi-hop

Evidence A:

Nina says she wants to reduce plastic waste in her neighborhood.

Evidence B:

Nina later repeats that reducing plastic waste is important to her.

Evidence C:

Nina again says that her main environmental goal is cutting plastic waste.

Bad question:

What environmental goal matters most to Nina?

Why this candidate must be rejected:

All selected evidence repeats the same fact. Multiple turns do not create multiple necessary reasoning steps.

Failure type:

repeated_evidence_pseudo_multihop

---

### Negative Example 3: The question reveals one reasoning step

Evidence A:

Daniel previously organized emergency food deliveries during a flood.

Evidence B:

A later volunteer role requires experience coordinating emergency supplies.

Bad question:

Which emergency-supply role is Daniel qualified for because he organized food deliveries during a flood?

Why this candidate must be rejected:

The question itself explicitly supplies the connection between Daniel's earlier experience and the later requirement. The reader is not required to discover the relationship.

A better question would ask:

Which earlier experience makes Daniel suitable for the later volunteer role?

Failure type:

question_leaks_intermediate_relation

---

## 5. Diversity requirements

Do not generate all candidates using the same pattern.

Across the candidate set, use multiple reasoning types when the dialogue supports them.

Possible types include:

* background_requirement_matching
* prior_evidence_later_choice
* source_verification_synthesis
* goal_action_alignment
* constraint_intersection
* multi_person_contribution
* comparison
* consistency_check
* causal_combination
* temporal_evolution
* prior_result_current_solution
* prediction_outcome_alignment

These labels are guidance, not fixed templates.

Do not force a reasoning type when the dialogue does not support it.

---

## 6. Output requirements

Generate exactly four candidate questions.

Each candidate must use at least:

* one parent-dialogue turn; and
* one child-dialogue turn.

Use no more than:

* three parent turns;
* four child turns.

Return only valid JSON with this structure:

```json
{
  "candidates": [
    {
      "candidate_id": "Q1",
      "reasoning_type": "background_requirement_matching",
      "parent_turn_aliases": ["P001"],
      "child_turn_aliases": ["C001"],
      "question": "A natural English open-ended question",
      "answer": "The complete canonical answer",
      "parent_contribution": "The necessary information supplied by the parent evidence",
      "child_contribution": "The necessary information supplied by the child evidence",
      "why_parent_only_is_insufficient": "A concise explanation",
      "why_child_only_is_insufficient": "A concise explanation",
      "why_combined_is_sufficient": "A concise explanation",
      "single_turn_shortcut_check": "State whether any selected turn contains the complete answer",
      "repeated_evidence_check": "State whether the selected evidence is redundant"
    }
  ]
}
```

Do not generate answer options or distractors.

Do not cite turns merely because they are related to the same topic.

Do not treat your own insufficiency explanations as proof. Select evidence conservatively.

Prefer producing fewer strong semantic relationships over creating artificial complexity, while still returning the required four candidates.

Before finalizing each candidate, verify:

* parent-only cannot uniquely answer it;
* child-only cannot uniquely answer it;
* parent and child together can uniquely answer it.
