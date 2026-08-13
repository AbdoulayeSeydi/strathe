with open("calibration/phase2_config.py", encoding="utf-8") as f:
    content = f.read()

old = '''    B_TRAINED_POLICY_BEHAVIOR = "CLAIM_B: trained PPO policy behavior under calibrated vs. default physics \u2014 complementary, not a second accuracy proof (no live hardware yet \u2014 see Phase 2.5)"'''

new = '''    B_TRAINED_POLICY_BEHAVIOR = "CLAIM_B: trained PPO policy behavior under calibrated vs. default physics \u2014 complementary, not a second accuracy proof (no live hardware yet \u2014 see Phase 2.5)"

    @staticmethod
    def b_trained_policy_behavior(algorithm: str = "PPO") -> str:
        """Same message as B_TRAINED_POLICY_BEHAVIOR, but with the real
        algorithm name substituted in - B_TRAINED_POLICY_BEHAVIOR itself
        is kept unchanged (hardcoded \"PPO\") for backward compatibility
        with anything that may already reference the bare constant, but
        train_multiseed.py now calls this instead, so a SAC/TD3 run's
        printed header and saved report both say the real algorithm that
        actually ran, not always \"PPO\" (a real, confirmed bug found and
        fixed - see PROJECT_STATUS.md)."""
        return (f"CLAIM_B: trained {algorithm.upper()} policy behavior under calibrated "
                f"vs. default physics \u2014 complementary, not a second accuracy proof "
                f"(no live hardware yet \u2014 see Phase 2.5)")'''

count = content.count(old)
print("Occurrences:", count)
if count == 1:
    content = content.replace(old, new, 1)
    with open("calibration/phase2_config.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("phase2_config.py updated successfully.")
else:
    print("REFUSING - count mismatch, do not proceed blindly.")