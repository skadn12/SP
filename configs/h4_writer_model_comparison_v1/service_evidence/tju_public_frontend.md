# TJU public frontend evidence

Scope: unauthenticated public static JavaScript GET only. No chat/completions, tokenizer, models, authentication or usage API was called. No key used. This is frontend evidence, not a verified v3 service contract.

## Confirmed public API sample
Source: https://ai.tju.edu.cn/assets/index-BAENSCav.js

Exact excerpts:
- `br="https://ai.tju.edu.cn/api/v3/",vr="tju-llm"`
- `curl '${br}chat/completions'` with `Content-Type: application/json`, `Authorization: Bearer <YOUR_API_KEY>`, and `{"model":"${vr}","messages":[{"role":"user","content":"你好"}]}`
- UI title: `OpenAI 兼容接口根路径`

This supports the provided endpoint, alias and Bearer format. It does not declare optional parameters or version pinning.

## Separate website chat implementation (v1, not v3)
Source: https://ai.tju.edu.cn/assets/ChatsView-B4fCc1-Q.js

Exact excerpts:
- `{name:"Qwen3.6-35B-A3B",value:"qwen3.6-35b-a3b"}`; the same model menu also includes Qwen3.6-27B and DeepSeek-V4-Flash.
- `new we({apiKey:v.value,apiBaseUrl:k+"/api/v1",systemMessage:"",maxModelTokens:1e5,maxResponseTokens:5e4,completionParams:{detail:!0,use_internet:L.value}})`
- `let o=g.value;q.value&&!$.value&&(o=o+"-no-think")`
- `sendMessage(e,{completionParams:{chatId:x.value,model:o,agent:T.value},systemMessage:"",...})`

The website has a thinking toggle implemented by changing the v1 model name. This does NOT establish that the v3 `tju-llm` alias accepts a `-no-think` alias or `enable_thinking` parameter. The client constants 100000/50000 are NOT server context/output limits and must not be copied into the experiment as such.

## Website client library details
Source: https://ai.tju.edu.cn/assets/index-Cob5bEGR.js

Exact excerpts:
- `this._completionParams={model:XM,temperature:.8,top_p:1,presence_penalty:1,...a}`
- `b={max_tokens:d,...this._completionParams,...l,messages:h,stream:a}`
- `function GM(t){return t.split("")}`
- `async _getTokenCount(e){return e=e.replace(/<\|endoftext\|>/g,""),GM(e).length}`

The website v1 client sends temperature and max_tokens. Its input counter is a JavaScript string split, not a model tokenizer; this does NOT validate experiment token counts or the v3 service's optional parameters.

## Unresolved for v3 tju-llm
- Fixed underlying model/revision. Public chat UI is consistent with the user's model name but does not prove alias routing.
- Actual server context window and maximum output.
- Whether max_tokens bounds visible output only or output plus reasoning.
- Reasoning default, supported controls and token accounting.
- temperature, response_format=json_object, seed, stop and other optional parameter support/semantics.
- Official tokenizer/chat template and per-message overhead.
- Prices, quota, timeout and HTTP retry contract.

No linked parameter document was found among the entry asset and its directly referenced JavaScript imports. Model API requests remain 0.
