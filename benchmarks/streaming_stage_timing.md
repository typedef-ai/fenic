# Sliding-window streaming stage timing

## Result

The fixed sliding window spends most of its time waiting for the next ordered
response, not admitting or advancing requests. The measurements refute the
hypothesis that per-window admission overhead dominates small-request
throughput.

At window 32, ordered slot wait accounts for 39.433 seconds, or 97.81% of the
40.316-second streaming median. Admission, synchronous dispatch, and window
advance total 0.959 seconds, or 2.38%. At window 100, slot wait falls to 16.229
seconds even though admission-side work rises to 2.013 seconds.

The dominant cost is therefore **ordered slot wait behind the fixed admission
window**. Increasing the window from 32 to 100 cuts streaming wall time by
21.605 seconds. Slot wait falls by 23.204 seconds while admission-side work
increases by 1.054 seconds. That direction is incompatible with admission
overhead causing the smaller window's regression.

## Measurement

The run used `gpt-4.1-nano` on 1,000 unique `semantic.map` inputs. It compared
standard execution with opt-in streaming at windows 32 and 100. Each cell ran
three times in a fresh child process with a fresh local database and response
cache. Arms were interleaved within each repetition. Client limits were 10,000
requests per minute and 10,000,000 tokens per minute. No run recorded a rate
limit event, and every run made exactly 1,000 physical requests.

The measured stack starts at
`a91d8af4a590ef9fb1f84338dbccf6c736378f56`. Instrumentation commit
`06870502030f0c876b364c9e32e49ef3af280210` adds stage durations to the
existing request lifecycle collector. It does not add another observation
mechanism.

Times below are medians of three runs. Stage values are medians of each run's
stage total. The standard path reports no sliding-window stages by design.

| Arm       | Window | Wall s | Window admission s | Request dispatch s | Slot wait s | Response drain s | Window advance s | Admission-side share |
| --------- | -----: | -----: | -----------------: | -----------------: | ----------: | ---------------: | ---------------: | -------------------: |
| Standard  |     32 |  8.997 |                  — |                  — |           — |                — |                — |                    — |
| Streaming |     32 | 40.316 |              0.029 |              0.929 |  **39.433** |           0.0008 |           0.0009 |                2.38% |
| Standard  |    100 |  8.084 |                  — |                  — |           — |                — |                — |                    — |
| Streaming |    100 | 18.711 |              0.037 |              1.976 |  **16.229** |           0.0009 |           0.0009 |               10.76% |

Streaming regressed 348.1% at window 32 and 131.5% at window 100 in this run.
The three streaming wall times were 39.586–49.297 seconds at window 32 and
18.275–24.553 seconds at window 100. The direction matches the earlier
regression even though provider variance changed the magnitude.

The collector emitted 5,000 stage records in each streaming run and zero in
each standard control. A provider-free loop emitted 5,000 equivalent records
in a median 4.272 milliseconds. This is less than 0.03% of the faster streaming
cell and cannot explain the measured gap.

## Stage definitions

- Window admission covers input iteration and request-key preparation.
- Request dispatch covers request preparation and synchronous queue handoff.
- Slot wait covers the ordered future wait.
- Response drain covers completed-future and live-window cleanup.
- Window advance covers successor-slot bookkeeping after admission and dispatch.

## Interpretation and proposal

The fixed window behaves as a concurrency ceiling for this non-rate-limited
workload. Window 32 needs more response waves than window 100, so ordered slot
wait grows even though request admission itself gets cheaper. This also
explains why a workload with few requests relative to a window can avoid the
penalty: the first fill admits the whole workload. Large requests can also make
the fixed scheduling overhead negligible relative to provider time.

A follow-up performance change should decouple the admission watermark from
the semantic operator's request batch size. One candidate is to admit at least
the provider's safe burst capacity while retaining a bounded live working set.
That proposal requires a separate correctness, memory, and provider benchmark;
this measurement does not implement it.

The accepted matrix cost $0.1380. A one-request instrumentation smoke cost an
additional $0.0000115.
