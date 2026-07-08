# API 403 Diagnosis and Resume Result

## Result

The earlier API 403 blocker was caused by provider balance/quota. After recharge, the minimal graph-style chat completion returned HTTP 200 and guarded mining resume completed.

## Non-Sensitive Configuration

| field | value |
| --- | --- |
| base_url | `https://api.siliconflow.cn/v1` |
| configured_model | `Qwen/Qwen3-14B` |
| api_key_present | `true` |
| api_key_length | `51` |
| key_material_printed | `false` |

## Minimal Tests

| test | before recharge | after recharge |
| --- | ---: | ---: |
| GET `/models` | 200 | 200 |
| configured model listed | true | true |
| minimal graph-style `/chat/completions` | 403 balance/quota blocker | 200 ok |

## Resume Mining

- api_graph_403_count_prior_to_recharge: 60
- api_graph_403_count_after_recharge_resume: 0
- resumed_generated_valid_clean: 300/300/300
- frontier_pool_count: 73
- frontier_plus_near_miss_count: 174
- ordering_bonus_remained_frozen: True

No API key material was printed. No training, no train_mix, and no next round were run.
