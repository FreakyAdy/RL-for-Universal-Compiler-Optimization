"""Validate pass sequences against registry grammar."""

from __future__ import annotations

from dataclasses import dataclass

from rl_uco.passes.registry import PassAction, PassRegistry


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""
    pipeline: str = ""


class PassSequenceValidator:
    def __init__(self, registry: PassRegistry):
        self.registry = registry

    def validate(self, sequence: list[PassAction], ir_kind: str = "llvm") -> ValidationResult:
        if not sequence:
            return ValidationResult(False, "empty sequence")

        depth = 0
        max_adaptor_depth = 2

        for action in sequence:
            if action.pass_id == self.registry.stop_action_id or action.name == "STOP":
                break
            if action.kind == "stop":
                break
            if action.kind == "baseline":
                continue

            if ir_kind == "mlir":
                if action.pass_id not in self.registry.mlir_passes and action.kind != "baseline":
                    if action.pass_id not in self.registry.passes:
                        return ValidationResult(False, f"unknown MLIR pass: {action.name}")
            else:
                known = (
                    action.pass_id in self.registry.passes
                    or action.pass_id in self.registry.adaptors
                )
                if not known and action.kind != "baseline":
                    return ValidationResult(False, f"unknown pass: {action.name}")

            if action.kind == "adaptor":
                depth += 1
                if depth > max_adaptor_depth:
                    return ValidationResult(False, "adaptor nesting too deep")
                if not action.adaptor_inner:
                    return ValidationResult(False, f"adaptor {action.name} missing inner passes")
            else:
                depth = 0

        if ir_kind == "mlir":
            pipeline = self.registry.mlir_pipeline_string(sequence)
        else:
            pipeline = self.registry.pipeline_string(sequence)

        if len(pipeline) > 8192:
            return ValidationResult(False, "pipeline string too long")

        return ValidationResult(True, "", pipeline)
