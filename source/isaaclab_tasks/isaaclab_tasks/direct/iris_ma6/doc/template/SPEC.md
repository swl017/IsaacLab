# {{ModuleName}} Specification

**Module:** `{{MODULE_NAME}}`
**Status:** Draft v1
**Last Updated:** YYYY-MM-DD

---

## 1. Motivation and Design Philosophy

### 1.1 The Problem
<!-- What problem does this module solve? Why is it needed? -->

### 1.2 Design Philosophy
<!-- Key design decisions and trade-offs. -->

### 1.3 Architecture Overview
```
<!-- ASCII diagram showing component relationships -->
```

## 2. Mathematical Formulation

### 2.1 System Model
<!-- Equations, notation, variable definitions -->

### 2.2 Algorithm Details
<!-- Core algorithm description with equations -->

## 3. Implementation Details

### 3.1 Configuration
<!-- Cfg dataclass fields, defaults, tuning guidance -->

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| | | | |

### 3.2 Core Class
<!-- Purpose, key methods, internal state -->

## 4. Integration Points

### 4.1 Input Interface
<!-- Tensor shapes, value ranges, coordinate frames -->

| Input | Shape | Range | Source |
|-------|-------|-------|--------|
| | | | |

### 4.2 Output Interface
<!-- Tensor shapes, value ranges -->

| Output | Shape | Range | Consumer |
|--------|-------|-------|----------|
| | | | |

### 4.3 Dependencies
<!-- Upstream and downstream modules -->

### 4.4 Calling Contract
<!-- CRITICAL for stateful modules -->

| Method | Type | Frequency | Lifecycle Hook | Notes |
|--------|------|-----------|---------------|-------|
| | WRITE | Once/step | `_post_physics_step` | Idempotent via timestamp guard |
| | READ | Any | Any | Safe for multiple calls |
| | CONFIG | On change | `_get_rewards` | Curriculum control |

**Stateful invariants:**
<!-- Assumptions about calling order, exclusivity, idempotency -->

## 5. Validation and Testing

### 5.1 Unit Tests
<!-- Test categories and pass/fail criteria -->

### 5.2 Integration Tests
<!-- Which modules to test with, expected interactions -->

## 6. Known Limitations
<!-- Current constraints, edge cases, planned improvements -->

## 7. References
<!-- Papers, external documentation, related specs -->
