# Open-source publication boundaries

The copyright holder authorized MIT licensing of first-party Core source on
2026-09-16, including modification, redistribution, and commercial use. npm's
`private: true` prevents accidental npm publication; it does not restrict the
MIT license or require a private GitHub repository.

## Code is public; user data is not

Publish a clean, reviewed current-source snapshot, not the incubation Git history.
Preserve the original private repository and its uncommitted development work.
Never export `.git`, real `.env` files, account databases, conversation transcripts,
mission recordings, experiment output, training datasets, credentials, signing
keys, Runtime disks, or caches. Synthetic tests and documented example requests
are source fixtures, not copies of production accounts.

The historical `evidence/` directory contains real session text and is excluded
from the public snapshot. References to those historical manifests in development
documents do not make the records public and do not qualify the current models.
Retain the original evidence privately, without editing its hashes or verdicts.

Production Supabase table contents, users, subscriptions, usage records, memory,
and storage objects remain private. Schema migrations, RLS rules, and functions
can be open-source without disclosing their data. A deployment uses its own
credentials; a public checkout does not grant access to the maintainer's service.

## Software, assets, and trained weights are separate

The root MIT license applies to first-party source and documentation, not to
third-party dependencies or private user content. Keep upstream notices and
trademarks intact. See [Third-party notices](../THIRD_PARTY_NOTICES.md).

The copyright holder separately authorized School Map and My Drone under MIT on
2026-09-16. The [asset license grant](default-assets-license.md) binds that
permission to the two exact archive hashes. Their historical `NOASSERTION`
metadata and qualification receipts remain unchanged; the external grant supplies
the later license declaration without manufacturing new qualification evidence.
Only those two reviewed archives are exported. Other assets and locally generated
models still require independent permission, review, and admission.

Training code is MIT; that alone does not establish redistribution rights to
datasets, third-party pretrained weights, or a derived checkpoint. Optional
ImageNet initialization must remain explicit. Publish a model only with its
upstream terms, training-data permission, hashes, and current control-contract
qualification. Missing rights or qualification fail closed.

## Validation and release

Source review and secret scanning are complementary: a clean scanner report does
not establish privacy, copyright ownership, or flight safety. Scan the exact
publication candidate before the first public push. Never upload scanner secrets
or original private evidence as CI artifacts.

A public source snapshot is not a signed installer or a flight-qualified release.
Keep model admission, exact asset identity, current Runtime checks, and explicit
mission confirmation enabled. The product repository owns the updater channel;
source publication alone must not advance it.

The obsolete standalone `windows-installer.yml` was retired from the public
snapshot: it supplied none of the three mandatory qualified-resource inputs and
still uploaded `0.1.0` filenames although the builder emits `1.0.0`. Its first
public commit and the private incubation repository preserve it for recovery.
Use the local `scripts/build-autonomy-windows.ps1` only with explicit current
model, admission, and native-sensor inputs. Formal signing and update channels
belong to the DroneDream product repository, not a second Core release workflow.

`PUBLIC_SOURCE_SNAPSHOT.json` is the inventory of the initial raw file export,
not a live Git-tree manifest. Later Git commits and line-ending normalization can
change files; release provenance must bind the actual commit and built file hashes.
