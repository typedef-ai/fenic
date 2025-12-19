# Changelog

## [0.7.0](https://github.com/typedef-ai/fenic/compare/v0.6.0...v0.7.0) (2025-12-19)


### Features

* update google-genai for thinking level, add support for gemini-3-flash preview, fix pricing for gemini-3-pro ([#289](https://github.com/typedef-ai/fenic/issues/289)) ([cea180f](https://github.com/typedef-ai/fenic/commit/cea180f2a13ec2d27624dfb8e517e5f5aada17cb))


### Bug Fixes

* cloud session and interface respects skip_usage_summary ([#288](https://github.com/typedef-ai/fenic/issues/288)) ([7978ea5](https://github.com/typedef-ai/fenic/commit/7978ea5ef4b839aa22766c2ab0e45d7d767b52ce))
* remove gpt-5.1 codex since fenic doesn't currently support its responses API ([#281](https://github.com/typedef-ai/fenic/issues/281)) ([54f7ec8](https://github.com/typedef-ai/fenic/commit/54f7ec80f69cb3bf20ffd3ffaa006b8ebab70b23))


### Documentation

* Readme updates ([#283](https://github.com/typedef-ai/fenic/issues/283)) ([65fdc84](https://github.com/typedef-ai/fenic/commit/65fdc84e1d5194c9c0165e822f20ec6f6b5d4525))

## [0.6.0](https://github.com/typedef-ai/fenic/compare/v0.5.0...v0.6.0) (2025-12-04)


### Features

* add `with_columns` with parallel execution ([#246](https://github.com/typedef-ai/fenic/issues/246)) ([16aea46](https://github.com/typedef-ai/fenic/commit/16aea46ee5d685424187713dcb7282d965f73bfe))
* add array_* functions ([#243](https://github.com/typedef-ai/fenic/issues/243)) ([727074b](https://github.com/typedef-ai/fenic/commit/727074be674e91e6254d110adcde028e49606322))
* Add openai support for semantic parse_pdf ([#253](https://github.com/typedef-ai/fenic/issues/253)) ([e3f58cd](https://github.com/typedef-ai/fenic/commit/e3f58cd9389f4c61513d3d6bf004a1b7c06996e0))
* Add pdf_parsing to openrouter ([#257](https://github.com/typedef-ai/fenic/issues/257)) ([8935115](https://github.com/typedef-ai/fenic/commit/893511513acc2724e72bdb16c8657ed9f73420cc))
* Add support for claude opus 4.5 and haiku 4.5 ([#280](https://github.com/typedef-ai/fenic/issues/280)) ([e1b620e](https://github.com/typedef-ai/fenic/commit/e1b620e84099901cb17270443b4876702b077892))
* add support for gemini3 pro ([#279](https://github.com/typedef-ai/fenic/issues/279)) ([eb68769](https://github.com/typedef-ai/fenic/commit/eb68769d3256c9e2ec0762156d6dd3727a56088c))
* add support for gpt-5.1 ([#278](https://github.com/typedef-ai/fenic/issues/278)) ([88602ac](https://github.com/typedef-ai/fenic/commit/88602ac0ccb955aca33ff5477080475d50d99e3f))
* add support for pyspark equivalent `explode_outer`, `posexplode`/`posexplode_outer` (via explode_with_index) ([#249](https://github.com/typedef-ai/fenic/issues/249)) ([8b035ca](https://github.com/typedef-ai/fenic/commit/8b035cad615ef76dce0aa94ed344de7292fce8c5))
* add support for sum_distinct, count_distinct, approx_count_distinct ([#242](https://github.com/typedef-ai/fenic/issues/242)) ([538dfa7](https://github.com/typedef-ai/fenic/commit/538dfa7cc314bad1801af1aff9fe4347a9d7c69d))
* fenic agents ([#267](https://github.com/typedef-ai/fenic/issues/267)) ([03bda18](https://github.com/typedef-ai/fenic/commit/03bda186bc703354bf8306138f94f0ce275e3421))
* implement regexp_* text functions ([#247](https://github.com/typedef-ai/fenic/issues/247)) ([87ef261](https://github.com/typedef-ai/fenic/commit/87ef2613f2942a147afb4a98b969384c5e426c64))
* llm response caching ([#269](https://github.com/typedef-ai/fenic/issues/269)) ([54f2f3e](https://github.com/typedef-ai/fenic/commit/54f2f3e656f09798aefed4a9f19744119cde94b9))
* Make timeouts configurable in semantic LLM operations ([#270](https://github.com/typedef-ai/fenic/issues/270)) ([cbf16eb](https://github.com/typedef-ai/fenic/commit/cbf16eb0ab707676329100951bb65aecaeed0967))
* pdf parsing evaluation tool and test pipeline ([#260](https://github.com/typedef-ai/fenic/issues/260)) ([20f2801](https://github.com/typedef-ai/fenic/commit/20f280181b00e5b15aaee1bd363044a9cafbec41))
* tweak pdf parser for corner cases and add 120s demo ([#259](https://github.com/typedef-ai/fenic/issues/259)) ([7c7454a](https://github.com/typedef-ai/fenic/commit/7c7454af9ebd56a599a161abb392d7339eeb330e))


### Bug Fixes

* assumptions about model attributes returned by OpenRouter API ([#261](https://github.com/typedef-ai/fenic/issues/261)) ([ab260bc](https://github.com/typedef-ai/fenic/commit/ab260bce6ba85f92774cfe8717770f0b33c46f08))
* embedded image links in 120s demo notebooks ([#255](https://github.com/typedef-ai/fenic/issues/255)) ([d44974b](https://github.com/typedef-ai/fenic/commit/d44974b9c91dd811c1a68e72311cab7ed8ee5ced))
* gemini vertex 2.0 flash models don't support profiles ([#254](https://github.com/typedef-ai/fenic/issues/254)) ([c99b9ac](https://github.com/typedef-ai/fenic/commit/c99b9acb3a48cb7bc7c819c9b1483fda9d2878f7))
* pin lancedb to avoid breaking change 0.25.0 ([#276](https://github.com/typedef-ai/fenic/issues/276)) ([5e4e861](https://github.com/typedef-ai/fenic/commit/5e4e861731e91e948dbbec758272a377b5766366))
* Semantic parse should only set higher temperature for google models ([#277](https://github.com/typedef-ai/fenic/issues/277)) ([8c71da3](https://github.com/typedef-ai/fenic/commit/8c71da3603b9c363b1fb0d1809f23dba4b07ecc7))


### Documentation

* clarify limitations of lit ([#262](https://github.com/typedef-ai/fenic/issues/262)) ([f2170a5](https://github.com/typedef-ai/fenic/commit/f2170a5d060686d372eca4ce245c229dc36ec0c3))
* page for timestamps, timezones, and dates ([#258](https://github.com/typedef-ai/fenic/issues/258)) ([893e866](https://github.com/typedef-ai/fenic/commit/893e866c59e1c670be9029362e6d84a6e5094663))

## [0.5.0](https://github.com/typedef-ai/fenic/compare/v0.4.2...v0.5.0) (2025-10-06)


### Features

* add "Fenic in 120 Seconds" demo notebook series ([#169](https://github.com/typedef-ai/fenic/issues/169)) ([e015e01](https://github.com/typedef-ai/fenic/commit/e015e0172dc42577e2f7d853a20ce734bd014c1a))
* Add page chunking to semantic parse_pdf, refactor token estimation ([#219](https://github.com/typedef-ai/fenic/issues/219)) ([a9b8132](https://github.com/typedef-ai/fenic/commit/a9b813286f063c6d94819702feb445e12b5cbd7f))
* add PhysicalPlanOptimizer library and implement MergeDuckDBNodesRule ([#213](https://github.com/typedef-ai/fenic/issues/213)) ([8729360](https://github.com/typedef-ai/fenic/commit/8729360e50b5aa776f2e135f263df01f3930dc03))
* add sonnet 4.5 to model catalog ([#240](https://github.com/typedef-ai/fenic/issues/240)) ([8d797c2](https://github.com/typedef-ai/fenic/commit/8d797c24f865997f2200876a6fa2d39bcafe41a9))
* add tz support for timestamp type ([#232](https://github.com/typedef-ai/fenic/issues/232)) ([c6167d9](https://github.com/typedef-ai/fenic/commit/c6167d9e5206c56525e5f499c16b6dbfb70cbf60))
* reader.pdf_metadata for reading pdf metadata into a df ([#182](https://github.com/typedef-ai/fenic/issues/182)) ([5636b1a](https://github.com/typedef-ai/fenic/commit/5636b1a2ef05961b1350fd3308f1320ddba5d92d))
* revert "feat: split reader.docs() into json() and markdown() ([#179](https://github.com/typedef-ai/fenic/issues/179))" ([#195](https://github.com/typedef-ai/fenic/issues/195)) ([8d9f490](https://github.com/typedef-ai/fenic/commit/8d9f4904ea630e7910a80aac40949a416ff23472))
* semantic.pdf_parse and support for google models ([#191](https://github.com/typedef-ai/fenic/issues/191)) ([a23ead5](https://github.com/typedef-ai/fenic/commit/a23ead5809d132dab801996add13498b532027bb))
* split reader.docs() into json() and markdown() ([#179](https://github.com/typedef-ai/fenic/issues/179)) ([083d61a](https://github.com/typedef-ai/fenic/commit/083d61a56fc14979a77831841086c0f2f99781c9))
* support date & timestamp data types in Fenic ([#201](https://github.com/typedef-ai/fenic/issues/201)) ([4388dec](https://github.com/typedef-ai/fenic/commit/4388dec5bfbd8ceedc5e40c33f9e9b2509c070df))
* support openrouter as a model provider ([#171](https://github.com/typedef-ai/fenic/issues/171)) ([b4e974f](https://github.com/typedef-ai/fenic/commit/b4e974ff926591fc55ef55019cc2fe0b7d6b43bd))
* System Tools for fenic MCP servers ([#163](https://github.com/typedef-ai/fenic/issues/163)) ([aa644dc](https://github.com/typedef-ai/fenic/commit/aa644dc0ff335800360aa6efb3860ea9c521a727))
* use gemini native token counter ([#197](https://github.com/typedef-ai/fenic/issues/197)) ([2978cd4](https://github.com/typedef-ai/fenic/commit/2978cd46a6f7974cfa1694d1dcafc989901d3f2d))


### Bug Fixes

* do not pass port to http_app, it is not accepted there ([#244](https://github.com/typedef-ai/fenic/issues/244)) ([fd11b30](https://github.com/typedef-ai/fenic/commit/fd11b30952511f3ffaeac1e012f01fcc3e7c7a35))
* do not retry on 429: quota exhausted errors from OpenAI ([#231](https://github.com/typedef-ai/fenic/issues/231)) ([deef3a6](https://github.com/typedef-ai/fenic/commit/deef3a6c797ac22237ac9bd60417a8c8d407bb07))
* doc loader should cast content in physical node and small api fixes ([#220](https://github.com/typedef-ai/fenic/issues/220)) ([3821a67](https://github.com/typedef-ai/fenic/commit/3821a671581006dc1b93083ebd3068187e3cce2c))
* enforce that return types of all branches in conditional chains are equal and handle child exprs in IsNullExpr ([#190](https://github.com/typedef-ai/fenic/issues/190)) ([bd778e5](https://github.com/typedef-ai/fenic/commit/bd778e50419acceb0e44a3fa1f269e3f0acc2c7d))
* export OpenRouterLanguageModel ([#227](https://github.com/typedef-ai/fenic/issues/227)) ([59be61b](https://github.com/typedef-ai/fenic/commit/59be61bad8543988f139261ef2d6f7290cbe852e))
* gemini-2.5-pro default model without profiles ([#209](https://github.com/typedef-ai/fenic/issues/209)) ([755a319](https://github.com/typedef-ai/fenic/commit/755a319313c9c78c9bd709b80baae1886a981cdf))
* improve localcatalog locking strategy to lock only when required ([#218](https://github.com/typedef-ai/fenic/issues/218)) ([8123ad9](https://github.com/typedef-ai/fenic/commit/8123ad964b04a4769ad21b03bcca397aa1d8294a))
* off-by-one bug in random pdf content generation for testing ([#226](https://github.com/typedef-ai/fenic/issues/226)) ([adc0070](https://github.com/typedef-ai/fenic/commit/adc0070fedb10950838b9554d41f9f4c88edcd15))
* openrouter batch completions client was broken after changes to LMRequestMessages ([#248](https://github.com/typedef-ai/fenic/issues/248)) ([4342fa6](https://github.com/typedef-ai/fenic/commit/4342fa61f83d11964d75c4773bd0bb31dc139b0b))
* parse_pdf returns MarkdownType column ([#252](https://github.com/typedef-ai/fenic/issues/252)) ([94c5481](https://github.com/typedef-ai/fenic/commit/94c5481b95a342f1571f01090ce35e3650e2c9df))
* quota exhaused error message for openai ([#233](https://github.com/typedef-ai/fenic/issues/233)) ([a42bc19](https://github.com/typedef-ai/fenic/commit/a42bc1904ee602b133130266e2e5eaa8d4e4cc42))
* raise exception in user thread instead of shutting down when request above model tpm ([#228](https://github.com/typedef-ai/fenic/issues/228)) ([3147aae](https://github.com/typedef-ai/fenic/commit/3147aae8c7d3bf3bbfda80af135bbd0bc2858703))
* raise if a request can't be handled by max token capacity ([#210](https://github.com/typedef-ai/fenic/issues/210)) ([b7eee1d](https://github.com/typedef-ai/fenic/commit/b7eee1da608a603a1c5ceb4ef1f4c946dba965ac))
* regressions from duckdb 1.4.0 behavior changes, use polars methods instead of arrow, minor test fixes ([#239](https://github.com/typedef-ai/fenic/issues/239)) ([f4a5e73](https://github.com/typedef-ai/fenic/commit/f4a5e73387f988b8bb2202be8b5787d8dec6c84f))
* remove o1-mini since it doesn't support json_schema structured output ([#208](https://github.com/typedef-ai/fenic/issues/208)) ([dec6503](https://github.com/typedef-ai/fenic/commit/dec6503069a9e42d3dfc13ff1445f53004737eae))
* skip local gemini token counting tests if google-genai is not installed ([#214](https://github.com/typedef-ai/fenic/issues/214)) ([7bb85c6](https://github.com/typedef-ai/fenic/commit/7bb85c62db6930824072d3b490bc0e3ab3709bad))
* unit test relies on non-deterministic calls in reduce ([#234](https://github.com/typedef-ai/fenic/issues/234)) ([a8f5770](https://github.com/typedef-ai/fenic/commit/a8f5770a3a6a69cc28cb8efa8c3804b24cdca519))
* when querying files in public S3 buckets/HuggingFace Datasets, the user should not have to provide credentials ([#202](https://github.com/typedef-ai/fenic/issues/202)) ([3a99f79](https://github.com/typedef-ai/fenic/commit/3a99f79d81566fadcc0735efec8e47d2cd4f6327))


### Documentation

* fixed pyproject.toml and readme for the mcp example ([#198](https://github.com/typedef-ai/fenic/issues/198)) ([407de03](https://github.com/typedef-ai/fenic/commit/407de0324660c70eee700caab05ebba50d122965))
* Update index.md in docs ([#205](https://github.com/typedef-ai/fenic/issues/205)) ([883cb30](https://github.com/typedef-ai/fenic/commit/883cb3078ce87e4b6adced2a46630201b6327362))
* Update README.md ([#204](https://github.com/typedef-ai/fenic/issues/204)) ([0410f89](https://github.com/typedef-ai/fenic/commit/0410f89ee5a6170675a53f7bc6e928dda7a937a9))
* updated and simplified the docs MCP example ([#193](https://github.com/typedef-ai/fenic/issues/193)) ([623a085](https://github.com/typedef-ai/fenic/commit/623a085247bd83ddf9ff3f237658cb09bb25ea46))

## [0.4.2](https://github.com/typedef-ai/fenic/compare/v0.4.1...v0.4.2) (2025-09-04)


### Bug Fixes

* Add 'self' to model_provider abstract methods ([#183](https://github.com/typedef-ai/fenic/issues/183)) ([25e8319](https://github.com/typedef-ai/fenic/commit/25e83193ed3a70d6f31cf3ce85b4b6a709ac1b56))
* cloud execution save_ methods should get metrics from backend ([#184](https://github.com/typedef-ai/fenic/issues/184)) ([4e32dd5](https://github.com/typedef-ai/fenic/commit/4e32dd52d28f518796d39170df9f576022738f59))
* default openai reasoning models without profile need reasoning tokens ([#185](https://github.com/typedef-ai/fenic/issues/185)) ([e3a3e2a](https://github.com/typedef-ai/fenic/commit/e3a3e2aea29d03836bdd2af3c449363ee89eaecc))
* regression when using google provider and logger level DEBUG ([#181](https://github.com/typedef-ai/fenic/issues/181)) ([c2d3045](https://github.com/typedef-ai/fenic/commit/c2d30459319357f0c66426a250f259de7e4cb0f9))

## [0.4.1](https://github.com/typedef-ai/fenic/compare/v0.4.0...v0.4.1) (2025-09-03)


### Bug Fixes

* write one metrics row per query not per operator to local fenic_system.query_metrics system table ([#177](https://github.com/typedef-ai/fenic/issues/177)) ([a37979c](https://github.com/typedef-ai/fenic/commit/a37979ce0ecfef47d04e02d427a9946120faf9f6))

## [0.4.0](https://github.com/typedef-ai/fenic/compare/v0.3.0...v0.4.0) (2025-09-02)


### Features

* Add connector for reading huggingface scheme ([#157](https://github.com/typedef-ai/fenic/issues/157)) ([9854cff](https://github.com/typedef-ai/fenic/commit/9854cffe2fcda1ae846846854071ec85b303f66c))
* Add gpt-5 support ([#134](https://github.com/typedef-ai/fenic/issues/134)) ([65cbb07](https://github.com/typedef-ai/fenic/commit/65cbb07eef63bd756b7129dafa16acab744c4a7f))
* add support for async udfs (for tool calling) ([2fb185f](https://github.com/typedef-ai/fenic/commit/2fb185fe8ce86423abd332d8ece2666802be5f12))
* Add verbosity and minimal reasoning params for gpt5 ([#135](https://github.com/typedef-ai/fenic/issues/135)) ([1fe1d95](https://github.com/typedef-ai/fenic/commit/1fe1d95b04c0f1ae30c08204d8700d554d57c0ae))
* allow users to add descriptions to views and tables ([#160](https://github.com/typedef-ai/fenic/issues/160)) ([12b5575](https://github.com/typedef-ai/fenic/commit/12b5575e39e60e54927e5815e64dcdcaf2d01d99))
* create `none()` and `empty()` column functions to allow users to populate columns with null/empty values ([#132](https://github.com/typedef-ai/fenic/issues/132)) ([8fafca7](https://github.com/typedef-ai/fenic/commit/8fafca7f4a55eb58ea36d69c4df007c0b7484e8a))
* declarative tool creation ([#150](https://github.com/typedef-ai/fenic/issues/150)) ([863cd70](https://github.com/typedef-ai/fenic/commit/863cd70c3f9d979b84988ad18defb9204d242fb8))
* local metrics table ([#161](https://github.com/typedef-ai/fenic/issues/161)) ([314e20f](https://github.com/typedef-ai/fenic/commit/314e20fca1bd81f554dd2a73df33f9cd3d4869af))
* make schema mismatch errors for union() more helpful ([#131](https://github.com/typedef-ai/fenic/issues/131)) ([d9087bd](https://github.com/typedef-ai/fenic/commit/d9087bdeb36570ae73c3695cb8c8c544ea734eec))
* MCP server example ([#120](https://github.com/typedef-ai/fenic/issues/120)) ([1169d7d](https://github.com/typedef-ai/fenic/commit/1169d7dbf5199e29e6b2667607e9efe96ff680a0))
* support claude-opus-4-1 ([#137](https://github.com/typedef-ai/fenic/issues/137)) ([2e74871](https://github.com/typedef-ai/fenic/commit/2e74871c9e87917d9e216d253c252aa8917f799f))
* support loading directory content into a dataframe ([#151](https://github.com/typedef-ai/fenic/issues/151)) ([83e151b](https://github.com/typedef-ai/fenic/commit/83e151b1bee0b73c315f81a966c059f2a83531f5))
* support logical types in cloud catalog  ([#140](https://github.com/typedef-ai/fenic/issues/140)) ([4e05c7c](https://github.com/typedef-ai/fenic/commit/4e05c7c884835a7987a7bdcd5d5e61076fe5b065))
* validate provider keys ([#164](https://github.com/typedef-ai/fenic/issues/164)) ([4cc5c72](https://github.com/typedef-ai/fenic/commit/4cc5c723ebb953b7604b1544794832f7f4b9f694))


### Bug Fixes

* add retry to OpenAI for 404 errors when no processing was done ([#176](https://github.com/typedef-ai/fenic/issues/176)) ([720f595](https://github.com/typedef-ai/fenic/commit/720f5952b3624cf91ba150a7a0b08852d3d6fc88))
* make local catalog threadsafe ([#156](https://github.com/typedef-ai/fenic/issues/156)) ([7f309b2](https://github.com/typedef-ai/fenic/commit/7f309b22026df7d2e677ce733349969e30139bac))
* make the error message when trying to load files from S3 without credentials set up more clear ([#165](https://github.com/typedef-ai/fenic/issues/165)) ([fb8f148](https://github.com/typedef-ai/fenic/commit/fb8f148abf9da5b5e490c3347dc8b2bda9b5e823))
* model_registry now uses the new ResolvedModelAlias for embedding model lookup properly ([#142](https://github.com/typedef-ai/fenic/issues/142)) ([3d80cd6](https://github.com/typedef-ai/fenic/commit/3d80cd6ee864d21010f213d6ed5a5f7d3205c72e))
* only use regex search for the mcp server example ([#148](https://github.com/typedef-ai/fenic/issues/148)) ([8fbbf72](https://github.com/typedef-ai/fenic/commit/8fbbf7288f0425a94d17c2db318778004fd0a3d5))
* shutdown event loop and cancel tasks on program exit ([#167](https://github.com/typedef-ai/fenic/issues/167)) ([b92dd1b](https://github.com/typedef-ai/fenic/commit/b92dd1b1b9c294bb76db1467555d88e2060d1e3f))
* test failure in lit(none) that somehow got merged ([#139](https://github.com/typedef-ai/fenic/issues/139)) ([73008c9](https://github.com/typedef-ai/fenic/commit/73008c9b608f75042086adf9f0a452722d87786c))
* use rust regex to validate regular expression used for rlike, ilike, like ([#162](https://github.com/typedef-ai/fenic/issues/162)) ([68f27b6](https://github.com/typedef-ai/fenic/commit/68f27b6153823a0b2f0f30ba4b964a8baeacbce8))


### Documentation

* enrich mcp server example README.md ([#145](https://github.com/typedef-ai/fenic/issues/145)) ([e0cd34f](https://github.com/typedef-ai/fenic/commit/e0cd34fb4391ca99e3ece6315133390667d6d269))
* fix group by docstring to use count() within agg() ([#152](https://github.com/typedef-ai/fenic/issues/152)) ([aaca9da](https://github.com/typedef-ai/fenic/commit/aaca9daf6d67d8785fc49d1eb9d0f2d22cc24829))
* Main readme colab ([#146](https://github.com/typedef-ai/fenic/issues/146)) ([d3842b0](https://github.com/typedef-ai/fenic/commit/d3842b0defc1283ede02afb2238ef5c1bcaaac37))
* Update example notebooks ([#144](https://github.com/typedef-ai/fenic/issues/144)) ([3cc0446](https://github.com/typedef-ai/fenic/commit/3cc044614c6203dfb7f0a354a382e9d3a416552a))
* update readme to reflect updated clustering api ([#127](https://github.com/typedef-ai/fenic/issues/127)) ([16c1ada](https://github.com/typedef-ai/fenic/commit/16c1ada9bdea9b17b499524e31c950c6d721345f))

## [0.3.0](https://github.com/typedef-ai/fenic/compare/v0.2.1...v0.3.0) (2025-08-04)


### Features

* Add basic support for WebVTT format to transcript parser ([97162af](https://github.com/typedef-ai/fenic/commit/97162afc50e6c56cc3add7efdb65907bb766d8ab)), closes [#71](https://github.com/typedef-ai/fenic/issues/71)
* Add full support for complex Pydantic models in `semantic.extract`; deprecate custom `ExtractSchema` ([#66](https://github.com/typedef-ai/fenic/issues/66)) ([b69baff](https://github.com/typedef-ai/fenic/commit/b69baffd6a991a769158f34945eca093457f9c96))
* add implementation for text.jinja() column function ([#98](https://github.com/typedef-ai/fenic/issues/98)) ([b784181](https://github.com/typedef-ai/fenic/commit/b78418155f5831133b7b28e338f41d48d83e97c5))
* add jinja expr and validate jinja template against input exprs ([#96](https://github.com/typedef-ai/fenic/issues/96)) ([4e71293](https://github.com/typedef-ai/fenic/commit/4e71293552a544854338dcca3bd4a504b119f984))
* Add jinja template validation and variable extraction to be used in jinja rendering column function and more ([#87](https://github.com/typedef-ai/fenic/issues/87)) ([fdd87e5](https://github.com/typedef-ai/fenic/commit/fdd87e548d16a23a109d630683c8576f23e9c886))
* add Jinja templating support to semantic operations ([9bcccff](https://github.com/typedef-ai/fenic/commit/9bcccff0a32372ba2a0a34a4443f4fdc3857a0ec))
* add pretty printed string representation for schema ([616f541](https://github.com/typedef-ai/fenic/commit/616f54104d897f0968939223071065b328e80495))
* add rust utility to convert arrow array values into minijinja contexts ([#97](https://github.com/typedef-ai/fenic/issues/97)) ([9901253](https://github.com/typedef-ai/fenic/commit/99012535f5d61d7ed4cce2d5d875b208c4aaf9ee))
* add support for basic fuzzy string ratio column functions ([791e662](https://github.com/typedef-ai/fenic/commit/791e662c7f96cb0445a71f52fd6ec0c84d4272eb))
* Add support for Cohere embeddings ([#116](https://github.com/typedef-ai/fenic/issues/116)) ([bf004df](https://github.com/typedef-ai/fenic/commit/bf004dfd3d193fcc9b3586dd3a52c5a41b7e6313))
* add support for greatest/least column functions ([0aa0636](https://github.com/typedef-ai/fenic/commit/0aa06365cba31c531fcdfed913cb422789f0baa3))
* Add support for webvtt transcript format ([#105](https://github.com/typedef-ai/fenic/issues/105)) ([97162af](https://github.com/typedef-ai/fenic/commit/97162afc50e6c56cc3add7efdb65907bb766d8ab))
* convert json/markdown functions to ScalarFunction ([#55](https://github.com/typedef-ai/fenic/issues/55)) ([1d5be25](https://github.com/typedef-ai/fenic/commit/1d5be2535e98dbd9b9c6f2e583b305caf5544c26))
* convert semantic/embedding exprs to ScalarFunction ([#56](https://github.com/typedef-ai/fenic/issues/56)) ([ea7f6ca](https://github.com/typedef-ai/fenic/commit/ea7f6ca778674f959a058eacbf54558d52581c23))
* function registry for signature validation ([#53](https://github.com/typedef-ai/fenic/issues/53)) ([a33747f](https://github.com/typedef-ai/fenic/commit/a33747f5c9527eb79b12e28ce6355bf73ed9fcd6))
* Implemented embeddings client for gemini with preset support for variable output dimensionality ([#111](https://github.com/typedef-ai/fenic/issues/111)) ([2104bfe](https://github.com/typedef-ai/fenic/commit/2104bfea3b66a9e71eedfbae95c3275218759710))
* make SemanticConfig optional to support OLAP-only and partial semantic operations ([#100](https://github.com/typedef-ai/fenic/issues/100)) ([5f8e3cb](https://github.com/typedef-ai/fenic/commit/5f8e3cb45af83ce62a4b99aa64dbffe323d0223a))
* register/convert text functions using ScalarFunction ([#54](https://github.com/typedef-ai/fenic/issues/54)) ([15a8c26](https://github.com/typedef-ai/fenic/commit/15a8c2647ba31adc743248719e7d48cb130a9705))
* replace pylance kmeans implementation with scikit-learn and expose num_init and max_iter params in api ([#104](https://github.com/typedef-ai/fenic/issues/104)) ([fdea6af](https://github.com/typedef-ai/fenic/commit/fdea6afc358935f6886c58c18095198a0f390a50))
* semantic map can now generate content with pydantic schema ([#78](https://github.com/typedef-ai/fenic/issues/78)) ([0148c81](https://github.com/typedef-ai/fenic/commit/0148c8155631ba52dcaf1f02e7ab9b3aee76ae34))
* summarization function ([#37](https://github.com/typedef-ai/fenic/issues/37)) ([2e83645](https://github.com/typedef-ai/fenic/commit/2e8364547910439544c7f37b16f70fc0311d0e57))
* support descriptions of class labels in semantic.classify ([baf3897](https://github.com/typedef-ai/fenic/commit/baf38971415844a919b674a97d4e6050b52ca831))
* support dynamic array index via expr arg in column.get_item() ([7795497](https://github.com/typedef-ai/fenic/commit/779549733c9c130cf3d7c1e3b5eeebe3a72c1640))
* Support for persistent views ([#41](https://github.com/typedef-ai/fenic/issues/41)) ([63747c0](https://github.com/typedef-ai/fenic/commit/63747c0cbfb0906248f20d6954a57cccc297b089))
* support model profiles with different thinking/reasoning configurations ([#82](https://github.com/typedef-ai/fenic/issues/82)) ([4a05c1a](https://github.com/typedef-ai/fenic/commit/4a05c1a4248781df3326949437f9cc260347e90c))
* use composition instead of inheritance for signature validation ([#63](https://github.com/typedef-ai/fenic/issues/63)) ([1b8983e](https://github.com/typedef-ai/fenic/commit/1b8983eb1bbfea515c63beb5aaf556400aadc153))


### Bug Fixes

* embedding profile assumptions ([#125](https://github.com/typedef-ai/fenic/issues/125)) ([131db13](https://github.com/typedef-ai/fenic/commit/131db13df6c69e8143994b972bd2e70e239db894))
* ensure total_output_tokens is populated even if the api response does not include it ([#62](https://github.com/typedef-ai/fenic/issues/62)) ([496c5fd](https://github.com/typedef-ai/fenic/commit/496c5fd1d17f9fd84ceb74bd863ac61d095b9eea))
* fix broken typesignature/functionsignature imports ([#84](https://github.com/typedef-ai/fenic/issues/84)) ([2a3460b](https://github.com/typedef-ai/fenic/commit/2a3460ba66cc93f785745265833a930f72409d31))
* fix bugs in text.extract related to delimiter sequence parsing and also disallow empty column names in templates ([#75](https://github.com/typedef-ai/fenic/issues/75)) ([fb96b13](https://github.com/typedef-ai/fenic/commit/fb96b13fcb7cf99d4625f9a6a5b481d179bb2d6e))
* grpc should use fenic's asyncio event loop ([#85](https://github.com/typedef-ai/fenic/issues/85)) ([356086a](https://github.com/typedef-ai/fenic/commit/356086ac0ac40d344833b5227afb190efae8e146))
* make catalog use sc method to implement does_table_exist ([#90](https://github.com/typedef-ai/fenic/issues/90)) ([513da9a](https://github.com/typedef-ai/fenic/commit/513da9a4f6f7fae58873df5bc233bd09319432f9))
* One language model param for test suites, with separate embeddings param ([#122](https://github.com/typedef-ai/fenic/issues/122)) ([7b3a297](https://github.com/typedef-ai/fenic/commit/7b3a2976d60bb9f3645590c7e7dcb2e3c020b300))
* rogue replacement of `name` with `model_name` ([#124](https://github.com/typedef-ai/fenic/issues/124)) ([efb5fba](https://github.com/typedef-ai/fenic/commit/efb5fba65b2e7f3a6e698327042a63b60190ed86))
* split plan creation from validation ([#115](https://github.com/typedef-ai/fenic/issues/115)) ([1d7c388](https://github.com/typedef-ai/fenic/commit/1d7c388005bbe097bf510b8169c855c83a5a865e))
* suppress noisy gemini logs ([#76](https://github.com/typedef-ai/fenic/issues/76)) ([89ce84c](https://github.com/typedef-ai/fenic/commit/89ce84c4d8fd8b57ec792b72b071feac405bce6e))
* Validate udf does not return LogicalType ([#117](https://github.com/typedef-ai/fenic/issues/117)) ([c1854ba](https://github.com/typedef-ai/fenic/commit/c1854ba991053bc77ec12bb2771f7dd20de9affd))


### Documentation

* fix expired discord link in contributing.md ([#73](https://github.com/typedef-ai/fenic/issues/73)) ([fe6fd3d](https://github.com/typedef-ai/fenic/commit/fe6fd3d6621a05814848d7b9cca54536e0336139))
* notebook version of the example ([#60](https://github.com/typedef-ai/fenic/issues/60)) ([6f2ad91](https://github.com/typedef-ai/fenic/commit/6f2ad91af9c82aa3085c901cc064b657ceb42cd7))

## [0.2.1](https://github.com/typedef-ai/fenic/compare/v0.2.0...v0.2.1) (2025-07-04)


### Bug Fixes

* support google models in cloud ([#52](https://github.com/typedef-ai/fenic/issues/52)) ([9031970](https://github.com/typedef-ai/fenic/commit/903197009fd5c4d9dfa8da086d40479e08d2b3e6))

## [0.2.0](https://github.com/typedef-ai/fenic/compare/v0.1.0...v0.2.0) (2025-07-03)


### Features

* migrate to google-genai package, support google vertex ([#35](https://github.com/typedef-ai/fenic/issues/35)) ([098f77d](https://github.com/typedef-ai/fenic/commit/098f77d515b70ddb0d64930dbdab76df425559b9))


### Bug Fixes

* add a grpc retry policy for the cloud engine ([#45](https://github.com/typedef-ai/fenic/issues/45)) ([8eb4cca](https://github.com/typedef-ai/fenic/commit/8eb4cca25bfc2f4464085c478f0a2328abfcfde4))
* add the remaining metrics to fix the error message when running a cloud execution ([#50](https://github.com/typedef-ai/fenic/issues/50)) ([cc9f8ca](https://github.com/typedef-ai/fenic/commit/cc9f8caa6e13374a375e63621e35e68fa39d01ea))
* increase the grpc default message size ([#51](https://github.com/typedef-ai/fenic/issues/51)) ([319d4d9](https://github.com/typedef-ai/fenic/commit/319d4d9ad89442fffe60e4cf4570da6da0c882c8))
* use basic types for expressions ([#48](https://github.com/typedef-ai/fenic/issues/48)) ([9e4b1af](https://github.com/typedef-ai/fenic/commit/9e4b1afa4c3272240f6e5ef9d43d7e495dd28c75))
* user should not have to provide admin secret for cloud session ([#49](https://github.com/typedef-ai/fenic/issues/49)) ([c608c6e](https://github.com/typedef-ai/fenic/commit/c608c6e5b0b08842b54c10236a57e6888f0a15fd))


### Documentation

* adding missing notebooks for examples ([#46](https://github.com/typedef-ai/fenic/issues/46)) ([81517f1](https://github.com/typedef-ai/fenic/commit/81517f1637b23d225d4b88d0e861549581181796))

## [0.1.0](https://github.com/typedef-ai/fenic/compare/v0.0.3...v0.1.0) (2025-06-25)


### Features

* add first and stddev aggregation functions ([#39](https://github.com/typedef-ai/fenic/issues/39)) ([86a142f](https://github.com/typedef-ai/fenic/commit/86a142f67a865600a80f13dc38b1830c8493cc85))
* add simple avg() aggregate function for the embedding type ([#33](https://github.com/typedef-ai/fenic/issues/33)) ([e467841](https://github.com/typedef-ai/fenic/commit/e467841ef6f4f4e950373d502f2cc28521534dab))
* Consume query metrics ([#32](https://github.com/typedef-ai/fenic/issues/32)) ([660ebae](https://github.com/typedef-ai/fenic/commit/660ebae4c227bf06c5896f6317b767be8111a5b4))
* make distance column name configurable in semantic.sim_join. ([#40](https://github.com/typedef-ai/fenic/issues/40)) ([e297a8a](https://github.com/typedef-ai/fenic/commit/e297a8ab3e3e23d03624b236646f1c400dd662f6))
* replace `semantic.group_by` with new `semantic.with_cluster_labels()` API ([#34](https://github.com/typedef-ai/fenic/issues/34)) ([83820e4](https://github.com/typedef-ai/fenic/commit/83820e4f4f918e6d07442b7557850528a713ae16))


### Bug Fixes

* fix bug in session.sql() when handling columns with embedding types ([#38](https://github.com/typedef-ai/fenic/issues/38)) ([b6a57ce](https://github.com/typedef-ai/fenic/commit/b6a57cefdc9fe3ed44ff097b43bbbab1cec6efa6))
* make markdown.extract_header_chunks() resilient to empty inputs and fix language filtering logic in get_code_blocks() ([#42](https://github.com/typedef-ai/fenic/issues/42)) ([e5d631d](https://github.com/typedef-ai/fenic/commit/e5d631d4e76780379157bf2cba2faadda6edfe96))
* Update tests to use the unresolved config session ([#31](https://github.com/typedef-ai/fenic/issues/31)) ([09cdc93](https://github.com/typedef-ai/fenic/commit/09cdc9364133b57c62b00fc09980e7efc815dc12))
* use the right entry points for GRPC calls to the cloud service ([#43](https://github.com/typedef-ai/fenic/issues/43)) ([bf43ec7](https://github.com/typedef-ai/fenic/commit/bf43ec7eaf51b756d22f2083a93c97847696587e))


### Documentation

* update discord link to a non-expired invite ([#27](https://github.com/typedef-ai/fenic/issues/27)) ([c4848e6](https://github.com/typedef-ai/fenic/commit/c4848e6c9d89f9539516b8ca38c74ac826c92dff))
* update discord link to match the one from website ([#29](https://github.com/typedef-ai/fenic/issues/29)) ([fd99722](https://github.com/typedef-ai/fenic/commit/fd9972296048b3bbb52ecf5567691ac9d6f84f2b))

## [0.0.3](https://github.com/typedef-ai/fenic/compare/v0.0.2...v0.0.3) (2025-06-19)


### Bug Fixes

* use SessionConfig for cloud execution ([#21](https://github.com/typedef-ai/fenic/issues/21)) ([e860720](https://github.com/typedef-ai/fenic/commit/e8607203685fd8b61b7ffb5584c182f0fb65cf1f))

## [0.0.2](https://github.com/typedef-ai/fenic/compare/v0.0.1...v0.0.2) (2025-06-18)


### Documentation

* use "supports" instead of "requires supports" in readme ([#22](https://github.com/typedef-ai/fenic/issues/22)) ([0500f42](https://github.com/typedef-ai/fenic/commit/0500f425493779d4ee655598a8083c7eb6de23b2))

## [0.0.1](https://github.com/typedef-ai/fenic/compare/v0.0.0...v0.0.1) (2025-06-18)


### Documentation

* add additional documentation for markdown -&gt; json conversion schema ([#17](https://github.com/typedef-ai/fenic/issues/17)) ([05c3921](https://github.com/typedef-ai/fenic/commit/05c39214a196de5e5177fe550003c8af490de152))
* add fenic logo to readme ([#20](https://github.com/typedef-ai/fenic/issues/20)) ([34c5704](https://github.com/typedef-ai/fenic/commit/34c57047d411808f1a7a8aa7ef76737ee06f68a3))
* add links to docs in github ([#12](https://github.com/typedef-ai/fenic/issues/12)) ([0718ce5](https://github.com/typedef-ai/fenic/commit/0718ce59641d50da0510b03db102fda0fa67eafa))

## Changelog
