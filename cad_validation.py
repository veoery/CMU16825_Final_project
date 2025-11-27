#!/usr/bin/env python3
"""
CAD Schema Validation Module

Provides validation functions for CAD JSON at different stages:
- Generation validation (before saving)
- Repair validation (quality checks)
- Export validation (pre-export checks)
"""

from typing import Tuple, List, Dict, Any


class CADValidator:
    """Validate CAD JSON structure and content"""

    @staticmethod
    def validate_extrude_heights(entities: Dict) -> Tuple[bool, List[str]]:
        """
        Check that all extrude features have non-zero heights.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        for ent_id, ent in entities.items():
            if ent.get("type") != "ExtrudeFeature":
                continue

            extent_one = ent.get("extent_one", {})
            distance_def = extent_one.get("distance", {})
            value = distance_def.get("value", 0)

            if value is None or value <= 0:
                issues.append(f"Extrude {ent_id}: invalid height {value} (must be > 0)")

        return len(issues) == 0, issues

    @staticmethod
    def validate_curve_completeness(entities: Dict) -> Tuple[bool, List[str]]:
        """
        Check that all curves have start_point and end_point.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        for ent_id, ent in entities.items():
            if ent.get("type") != "Sketch":
                continue

            profiles = ent.get("profiles", {})

            # Check for malformed profiles (list instead of dict)
            if isinstance(profiles, list):
                issues.append(f"Sketch {ent_id}: 'profiles' is a LIST (should be DICT) - Fusion 360 format detected")
                continue

            for prof_id, prof in profiles.items():
                loops = prof.get("loops", [])

                for loop_idx, loop in enumerate(loops):
                    curves = loop.get("profile_curves", [])

                    for curve_idx, curve in enumerate(curves):
                        if "start_point" not in curve:
                            issues.append(
                                f"Sketch {ent_id}/{prof_id}/loop[{loop_idx}]/curve[{curve_idx}]: "
                                f"missing start_point"
                            )
                        if "end_point" not in curve:
                            issues.append(
                                f"Sketch {ent_id}/{prof_id}/loop[{loop_idx}]/curve[{curve_idx}]: "
                                f"missing end_point"
                            )

        return len(issues) == 0, issues

    @staticmethod
    def validate_profile_curves(entities: Dict) -> Tuple[bool, List[str]]:
        """
        Check that all profiles have at least one curve in their loops.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        empty_profiles = []

        for ent_id, ent in entities.items():
            if ent.get("type") != "Sketch":
                continue

            profiles = ent.get("profiles", {})

            # Check for malformed profiles (list instead of dict)
            if isinstance(profiles, list):
                issues.append(f"Sketch {ent_id}: 'profiles' is a LIST (should be DICT) - Fusion 360 format detected")
                continue

            for prof_id, prof in profiles.items():
                loops = prof.get("loops", [])

                has_curves = False
                for loop in loops:
                    curves = loop.get("profile_curves", [])
                    if curves:
                        has_curves = True
                        break

                if not has_curves:
                    empty_profiles.append(f"{ent_id}/{prof_id}")
                    issues.append(
                        f"Sketch {ent_id}: profile {prof_id} has no curves (all loops empty)"
                    )

        return len(issues) == 0, issues

    @staticmethod
    def validate_extrude_profiles(entities: Dict) -> Tuple[bool, List[str]]:
        """
        Check that extrude features reference valid profiles with curves.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        for ent_id, ent in entities.items():
            if ent.get("type") != "ExtrudeFeature":
                continue

            profiles = ent.get("profiles", [])
            for prof_ref in profiles:
                sketch_id = prof_ref.get("sketch")
                profile_id = prof_ref.get("profile")

                if sketch_id not in entities:
                    issues.append(
                        f"Extrude {ent_id}: references missing sketch {sketch_id}"
                    )
                    continue

                sketch = entities[sketch_id]
                if sketch.get("type") != "Sketch":
                    issues.append(
                        f"Extrude {ent_id}: sketch {sketch_id} is not a Sketch"
                    )
                    continue

                sketch_profiles = sketch.get("profiles", {})
                if profile_id not in sketch_profiles:
                    issues.append(
                        f"Extrude {ent_id}: profile {profile_id} not found in sketch {sketch_id}"
                    )
                    continue

                # Check if profile has curves
                profile = sketch_profiles[profile_id]
                has_curves = False
                for loop in profile.get("loops", []):
                    if loop.get("profile_curves"):
                        has_curves = True
                        break

                if not has_curves:
                    issues.append(
                        f"Extrude {ent_id}: profile {profile_id} has no curves"
                    )

        return len(issues) == 0, issues

    @staticmethod
    def validate_all(entities: Dict, strict: bool = False) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Run all validations.

        Args:
            entities: Entities dict from CAD JSON
            strict: If True, reject any issues. If False, warn but allow.

        Returns:
            (is_valid, issues_dict) where issues_dict has keys:
            - extrude_heights
            - curve_completeness
            - profile_curves
            - extrude_profiles
        """
        all_issues = {}
        all_valid = True

        # Run all checks
        valid, issues = CADValidator.validate_extrude_heights(entities)
        all_issues["extrude_heights"] = issues
        if not valid:
            all_valid = False

        valid, issues = CADValidator.validate_curve_completeness(entities)
        all_issues["curve_completeness"] = issues
        if not valid:
            all_valid = False

        valid, issues = CADValidator.validate_profile_curves(entities)
        all_issues["profile_curves"] = issues
        if not valid:
            all_valid = False

        valid, issues = CADValidator.validate_extrude_profiles(entities)
        all_issues["extrude_profiles"] = issues
        if not valid:
            all_valid = False

        return all_valid, all_issues

    @staticmethod
    def print_validation_report(entities: Dict, strict: bool = False) -> bool:
        """
        Print validation report and return whether validation passed.

        Args:
            entities: Entities dict from CAD JSON
            strict: If True, treat warnings as errors

        Returns:
            True if valid, False otherwise
        """
        is_valid, all_issues = CADValidator.validate_all(entities, strict)

        has_issues = any(issues for issues in all_issues.values())

        if not has_issues:
            print("✅ All validations passed")
            return True

        print("\n⚠️  VALIDATION REPORT:")
        print("=" * 80)

        for check_name, issues in all_issues.items():
            if issues:
                print(f"\n❌ {check_name}:")
                for issue in issues:
                    print(f"   • {issue}")

        print("\n" + "=" * 80)

        if strict:
            print(f"❌ Validation FAILED (strict mode)")
            return False
        else:
            print(f"⚠️  {sum(len(i) for i in all_issues.values())} issue(s) found (non-strict)")
            return True


def generate_validation_report(json_data: Dict) -> Tuple[bool, str]:
    """
    Generate a validation report for generated CAD JSON.

    Suitable for use in generation notebooks to validate before saving.

    Args:
        json_data: Generated CAD JSON data

    Returns:
        (is_valid, report_str)
    """
    entities = json_data.get("entities", {})

    is_valid, all_issues = CADValidator.validate_all(entities, strict=True)

    report = []

    if is_valid:
        report.append("✅ VALIDATION PASSED - Safe to export")
        report.append(f"\nStatistics:")
        report.append(f"  Sketches: {sum(1 for e in entities.values() if e.get('type') == 'Sketch')}")
        report.append(f"  Extrudes: {sum(1 for e in entities.values() if e.get('type') == 'ExtrudeFeature')}")
    else:
        report.append("❌ VALIDATION FAILED - Do not export")
        report.append("\nIssues found:")

        for check_name, issues in all_issues.items():
            if issues:
                report.append(f"\n  {check_name}:")
                for issue in issues[:3]:  # Show first 3 issues per category
                    report.append(f"    • {issue}")
                if len(issues) > 3:
                    report.append(f"    ... and {len(issues) - 3} more")

    return is_valid, "\n".join(report)


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Validate CAD JSON")
    parser.add_argument("json_file", help="CAD JSON file to validate")
    parser.add_argument("--strict", action="store_true", help="Fail on any issues")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    is_valid = CADValidator.print_validation_report(data.get("entities", {}), strict=args.strict)
    exit(0 if is_valid else 1)
