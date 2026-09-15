# Wikidata Constraint Types — Brief Guide


## Typing (what kinds of things subjects/values may be)

### `type`

Constrains the **subject** that uses a property to be an instance/subclass of one or more specified classes.
*Example:* items with `date of birth (P569)` should be `human (Q5)` or similar living-being classes.

### `valueType`

Constrains the **value** of a property to be of certain classes (for item-valued properties) or datatypes (for literals).
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

All values of the property must be **globally unique across items** (no two items share the same value).
*Example:* external IDs like `VIAF ID (P214)` must be distinct.

---

## Consistency / Mutual exclusion

### `conflictWith`

Two properties (or specific value patterns) **should not co-occur** on the same item; if one is present, the other must be absent.
*Example:* `date of birth (P569)` conflicting with `year of birth missing (Q...)`-style markers, or mutually exclusive status flags.
