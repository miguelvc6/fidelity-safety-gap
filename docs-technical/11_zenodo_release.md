# Zenodo release preparation

The maintained software is published as a fixed GitHub release. The associated
Zenodo record contains the immutable data, model, result, and supplementary-text
artifacts needed for the paper.

## Archive boundary

The upload contains:

- the exact unlabeled benchmark rows used for symbolic evaluation;
- the exact labeled train, validation, and test rows used to materialize the
  training graphs;
- the frozen global integer encoder, constraint registry, fixed 2018 hierarchy,
  validator-v2 label manifest, graph manifests, and target vocabulary;
- the selected checkpoints for Direct--Passive GNN, Direct--Factor GNN,
  Candidate--C, Candidate--DP, and Candidate--SR;
- the tracked predictions, metric outputs, diagnostics, configurations, and
  training histories for the five learned systems and four deterministic
  baselines; and
- the supplementary results PDF and explicit artifact-licence files.

The generated graph payloads are not uploaded. They occupy 89.4 GB and can be
reconstructed from the archived inputs with the pinned graph builder. Duplicate
or transitional files, including `checkpoint.last.pth`, legacy reranker JSON,
pre-schema-v3 backups, plots, logs, and exploratory runs, are also excluded.

The published `v1.0.0` bundles are immutable historical artifacts. They must be
verified in place with `sha256sum --check SHA256SUMS` and never overwritten by
the validator-v2 regeneration. External publication of a new version is a
separate, out-of-scope release action.

## Build a candidate upload

From the repository root, run:

```bash
uv run python scripts/prepare_zenodo_release.py --release-version <new-version>
```

The output is written to `release/zenodo/<new-version>/`, which is ignored by Git. The
script requires all selected result files to be tracked and unchanged. It
creates deterministic `tar.gz` bundles, a per-file provenance manifest, an
upload README, and `SHA256SUMS`.

The version DOI reserved for the artifact record is
`10.5281/zenodo.22013512`. It is the script default and is recorded in both the
README and manifest. To state it explicitly, run:

```bash
uv run python scripts/prepare_zenodo_release.py --release-version <new-version> --doi <new-version-doi>
```

## Final checks before publication

Do not publish the Zenodo record until:

1. the release commit is clean and tagged with the version passed to the script;
2. `ARTIFACT_MANIFEST.json` reports that the tag was present and the worktree was
   clean;
3. the generated files pass `sha256sum --check SHA256SUMS`;
4. each archive can be listed and extracted without error;
5. the GitHub release is publicly accessible without authentication; and
6. the DOI and exact software release are present in the paper's
   `\supplementdetails` metadata.

Publishing a Zenodo version freezes its files. Subsequent changes should use a
new record version rather than replacing the paper's cited artifact version.
