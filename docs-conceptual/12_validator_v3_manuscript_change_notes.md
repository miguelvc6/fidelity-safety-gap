# Validator-v3 Manuscript Change Notes

The current manuscript text and empirical values predate validator semantics
v3 and must not be interpreted as results for the corrected pipeline. They are
left untouched until the complete artifact suite is regenerated and passes the
machine-readable readiness gates.

After regeneration, Table 1 and its surrounding explanation must describe:

- state-aware `P279` reachability over an immutable 1 July 2018 background plus
  local additions and deletion tombstones;
- separate direct-adjacency and closure completeness, with conservative
  `unknown` negatives;
- exact primary-ID binding after supported-family filtering;
- detailed unresolved-edit effects, including possible new secondary anchors;
- exceptions applied before primary vacuity and anchor-only distinct-values
  exemption scope; and
- the constrained-property and hierarchy-parser validation rules.

The methods section must distinguish Candidate--C/Candidate--DP's fixed
pre-violated proven-fix loss from Candidate--SR's satisfaction-only objective.
It must state that post-edit unknown candidates are not proven fixes, while
unknown symbolic labels remain unknown rather than being reclassified.

All result tables, abstract comparisons, discussion claims, diagnostics, and
limitations affected by validators, factor labels, candidate objectives, or
artifact populations must be replaced only from accepted schema-v3 outputs.
The limitations should explicitly retain the bounded triple abstraction,
`P4155` qualifier loss, incomplete historical ancestry, and the distinction
between the fixed hierarchy cutoff and individual edit timestamps.

The non-reference body must still fit within 12 pages after a conventional
LaTeX build, with warnings and overflow checked. No old numerical claim should
be edited piecemeal before the full suite passes.
