# v119 QA retry pilot

本目录是 v118 的独立、最多一次重试批次，包含 root 19708、94289、103866。

- v118 目录只读，本目录不覆盖、不回写。
- 只复制 baseline、continuation constraints、六个 READY source、B 事件输入和结构收据。
- 没有复制旧 QA/private asset design；尤其 root103866 不含 PRIVATE_ASSET_DESIGN、QA_DRAFT、QA_AUTHOR_PAYLOAD 或 solver 分数。
- 先保存 ordering anchors，再生成 QA；独立 reviewer 只能读取公开题目和完整 source。
- 每个 root 只允许一次 retry；成功结果不能直接并入 v118 或正式训练池。
