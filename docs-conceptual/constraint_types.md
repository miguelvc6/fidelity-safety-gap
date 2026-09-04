# Wikidata Constraint Types — Brief Guide


## Typing (what kinds of things subjects/values may be)

### `type`

Constrains the **subject** that uses a property according to the definition's
instance, subclass, or either selector. Class ancestry is evaluated against the
fixed 1 July 2018 hierarchy.
*Example:* items with `date of birth (P569)` should be `human (Q5)` or similar living-being classes.

### `valueType`

Constrains each represented **value** of a property with the analogous
instance/subclass/either class relation. Literal/datatype forms that the
benchmark cannot represent are `unknown`.
*Example:* values of `mother (P25)` must be items of class `human (Q5)`.

---

## Dependency (what else must be stated)

### `itemRequiresStatement`

If an item has a given property, it **must also** have another specified statement.
*Example:* if an item has `country of citizenship (P27) = Austria`, it must also have `place of birth (P19)`.

### `valueRequiresStatement`

For each **target value** used with a property, that **value item itself** must carry another statement.
*Example:* if `employer (P108)` points to organization *X*, then *X* must have `instance of (P31) = organization`.

### `inverse`

If item **A** uses property **P** to point to item **B**, then **B** should have the **inverse** property **P′** pointing back to **A**.
*Example:* `parent (P40)` on child implies `child (P40)` (or `has child`) on the parent item, depending on modeling.

### `symmetric`

If item **A** uses property **P** to point to item **B**, then **B** should also use the same property **P** to point back to **A**. This is evaluated as inverse-style evidence where the inverse property is the constrained property itself.

---

## Value domain / enumeration

### `oneOf`

Restricts the value to be **one of an explicit, finite list** of allowed items (a closed set).
*Example:* `sex or gender (P21)` value must be one of a curated list (as defined in the constraint).

---

## Cardinality & Uniqueness

### `single`

An item should have **at most one** statement for the property (no multiple values).
*Example:* `date of birth (P569)` is expected to appear once per item.

### `distinct`

No two represented subjects may share the same applicable value. This is a
bounded local check, not a claim of global Wikidata uniqueness.
*Example:* external IDs like `VIAF ID (P214)` must be distinct.

---

## Consistency / Mutual exclusion

### `conflictWith`

Two properties (or specific value patterns) **should not co-occur** on the same item; if one is present, the other must be absent.
*Example:* `date of birth (P569)` conflicting with `year of birth missing (Q...)`-style markers, or mutually exclusive status flags.
