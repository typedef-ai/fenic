# Changelog

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
