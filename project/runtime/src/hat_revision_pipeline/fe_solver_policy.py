"""Default FE solver and dependency-scoped cache revision."""
from __future__ import annotations

FE_SOLVER_REVISION = "fe-compact-task-specific-v3"
FE_SINGLE_OPTIMIZER = "L-BFGS-B"
FE_MULTITRAP_OPTIMIZER = "BFGS"
FE_EVALUATOR = "compact"


def is_fe_objective(objective):
    spec = getattr(objective, "method", None)
    return spec is not None and (
        getattr(spec, "alpha_per_m", 0.0) > 0.0 or
        str(getattr(spec, "name", "")).lower() in
        {"fe", "gfe", "fe+g", "force-equilibrium", "rh", "regularized-fe"})


def require_fe_solver(optimizer=None, evaluator_backend=None, *, task="Single"):
    """Enforce the declared compact FE solver for each problem size.

    Single-target FE/GFE/RH uses unbounded L-BFGS-B. Multi-target FE/GFE uses
    BFGS. Both use compact evaluation. Explicit incompatible choices fail
    before a command cache is read or optimization starts.
    """
    if task not in ("Single", "Triple", "Multi"):
        raise ValueError(f"Unknown FE solver task: {task!r}")
    expected = FE_SINGLE_OPTIMIZER if task == "Single" else FE_MULTITRAP_OPTIMIZER
    if optimizer not in (None, expected):
        raise ValueError(
            f"{task} FE/GFE/RH requires {expected}; received {optimizer!r}")
    if evaluator_backend not in (None, FE_EVALUATOR):
        raise ValueError(
            f"{task} FE/GFE/RH requires the compact evaluator; received {evaluator_backend!r}")
    return expected, FE_EVALUATOR


# These identities describe unchanged dependencies across this exact solver-only
# edit. Unknown future source revisions retain their own identity.
SOURCE_ID_COMPATIBILITY = {'000000000000343926d36f621da8f722': '000000000000346648590d8b02650376', '000000000000720ab63958a8f9ec1dae': '00000000000072081031d5e8121a1dbe', '0000000000001330b6524a93ef822330': '00000000000011fa76a9e6cfc59ec648', '00000000000076f3a1414545e9252465': '00000000000075d41a89495f8a75bb72', '000000000000bf8c853faff017eb3736': '000000000000b90813e3053280840e08', '00000000000167bc3e90222cdd3ed0a6': '00000000000166e6da5294621fbf87aa', '000000000002240bb6abcb32728a0c18': '00000000000223672f250d03dda2dbd2', '000000000000e3f2556cac0e8c4eba0f': '000000000000e257a2a532d001a23464', '00000000000019646b8fbd0397503498': '0000000000001960b702799dee523428', '0000000000007d98d3e3269ea19b87fc': '0000000000007cf6abf50cbce8fa5c7a', '0000000000003b79f9ec7512e5503700': '0000000000003b29d613e46534a91b66', '000000000000a95d7c2dca4b3c93a964': '000000000000a63e51bdac03e261aaa9', '0000000000001792c72247a0acc54483': '0000000000000fed95ac1c29d9721d73', '00000000000014f20e3e6065896b81bf': '000000000000141083284b7cf0a63471', '0000000000006246c92b09725b1205ff': '000000000000625aff999dd0b0780d2d', '000000000000461f1a57679ce92aa387': '00000000000044e5d87fa28c225639f5', '0000000000007729694ab00252a8b085': '000000000000735ce77dc2ed1bb977cc', '0000000000002cd616240a6bb299cc08': '0000000000002c199f99c719d43f9040', '0000000000021f36875f8cdcf2437650': '0000000000021e92cf2ec939a8594619', '000000000000e34681532edd058a7d51': '000000000000d3003de0cf648fd01dbd', '0000000000007d929721f4812411870b': '0000000000007cf04c35329881505b89', '000000000000363f446f3c6042c894d5': '00000000000035ef47f77c31d0b0793b', '00000000000014f6643c7a08ba4383a6': '0000000000001414d9b238b773813658', '000000000000366f85b5d298f372b002': '0000000000003394c782c6413af5c263', '000000000000bfbe6269df74e54448b2': '000000000000bde517a6633114aaa646', '00000000000079a66441be24c3360457': '00000000000075d41a89495f8a75bb72', '000000000000ee6b8a64948e7961cc49': '000000000000e500b3a03d2a8576f963', '00000000000060b0652116157eebe78d': '000000000000600cacad3a9ec657acef', '000000000000901e2ac8343d0ff9b310': '0000000000008e6af112cb6d18e11f64', '0000000000007ebd3dd83cc28ba2e60f': '0000000000007cf04c35329881505b89', '0000000000001662b0b9537c22ecf809': '0000000000001414d9b238b773813658', '000000000000e5d4b36db9880f8d5f36': '000000000000e257a2a532d001a23464', '000000000001090f1e0f8624bea918fd': '00000000000107f46baf659670dab711', '000000000000d85623b9cd8e6dbe66ec': '000000000000d7e9dc3292d414334623', '0000000000000659c3eddd575694fccf': '000000000000072a76ed738975b24215', '0000000000005ffa1eb5ea157f2aa9e4': '00000000000061247c2972f51b670b7a', '0000000000002e33789f34bc6c21410f': '0000000000002c199f99c719d43f9040', '000000000000785d1fad0a5478881160': '000000000000735ce77dc2ed1bb977cc', '00000000000032f846c09bfdc0c38100': '000000000000346648590d8b02650376', '00000000000074c6dc6d29ce82aedb77': '000000000000721a15a8ab7d76742386', '0000000000000bf436bb221c5b20af98': '0000000000000addc168ba018b724fca', '0000000000009861285a0a6d41463375': '0000000000008d586f4d1e2a29547df5', '0000000000008c3fe75ccd70c8ccde60': '0000000000009288a2a95ca00fd8eda4', '00000000000080e3736186fa2ab46f1d': '0000000000007f45338f00b8135be6c1', '0000000000007ec3febf5cf12890e700': '0000000000007cf6abf50cbce8fa5c7a', '000000000000ab2aebc4e75156298aba': '0000000000009c26344a30b4a658a618', '00000000000038d5e851e02268c62bd7': '00000000000037ec06810aeae1e6d24e', '000000000000aa7e899d971592eb0aca': '000000000000aa06a4ddfcb44faee05c', '000000000000165ed152dadfd90bf622': '000000000000141083284b7cf0a63471', '00000000000037be6166db82ea4ed385': '0000000000005852f460c462608d6c9d', '0000000000017305a887e81f422f9e2d': '00000000000166e6da5294621fbf87aa', '00000000000077b6867bc3d9a76d2a98': '00000000000077bf6f168e3ae77e3559', '000000000000677657435adf97f13038': '000000000000698aa4c83bde57dddd0d', '000000000000ab2438bf1f4000412947': '000000000000a63e51bdac03e261aaa9', '0000000000000a0d7f1a81cfd54428a8': '00000000000009225b60cd50b05edd3e', '00000000000019606128cb1220182eb1': '0000000000001960b702799dee523428', '00000000000018af8326279e32719ecb': '0000000000000fed95ac1c29d9721d73', '000000000000237b0457fabd12c06c09': '00000000000022ef8fffcdeabb0e42d1', '00000000000061da6978e3cace7edfea': '000000000000625aff999dd0b0780d2d', '00000000000036c2c5b24e82216bfc8a': '00000000000036b1c85e13056e4ff753', '0000000000004b6f1b0a8b6eafc4ffe8': '0000000000004b4aced3ed2f2075f153', '0000000000004f0e548461bc7925a990': '0000000000004ec38a191fecef1091bb', '00000000000084a5ec6c47d40a7f067b': '00000000000084a1c85be2a1aa270593', '0000000000017646cd198cf8fbe9848d': '00000000000166e6da5294621fbf87aa', '00000000000053f4708de16fc3429c96': '00000000000053dabdd0b7cb33ec9488', '0000000000002f1c0b19198c485d903c': '0000000000002c199f99c719d43f9040', '00000000000046760bb9dfa03a5fc1d5': '00000000000044e5d87fa28c225639f5', '0000000000008dd5a8720b48cd2c65a5': '0000000000009288a2a95ca00fd8eda4', '00000000000045ea3f35fc00d87f153b': '00000000000043bf98b0674bbaba2fce', '000000000000aa76f8ac0b58884a07ee': '000000000000aa06a4ddfcb44faee05c', '000000000001784defcad185efbc323d': '00000000000166e6da5294621fbf87aa', '00000000000037979895654b3f120ed6': '0000000000003394c782c6413af5c263', '000000000000bfccd7b2a9fb9f444de8': '000000000000bde517a6633114aaa646', '00000000000062054f388845bd49ef4f': '000000000000625aff999dd0b0780d2d', '000000000000ae3c089f10e5d9ca8f83': '0000000000009c26344a30b4a658a618', '000000000000e607f5619e2814876ffa': '000000000000e257a2a532d001a23464', '00000000000109eae31e49d287fd630c': '00000000000107f46baf659670dab711', '000000000000792366f4c0ee759d5ba8': '000000000000735ce77dc2ed1bb977cc', '00000000000074fc6a047bb6a3afedd5': '000000000000721a15a8ab7d76742386', '00000000000036ba31773dee905afaba': '00000000000036b1c85e13056e4ff753', '0000000000000bf480bb275d4203afb1': '0000000000000addc168ba018b724fca', '00000000000178c5373d087efcda5c11': '00000000000166e6da5294621fbf87aa', '000000000000ab924dcd1827b31f4c7b': '000000000000a63e51bdac03e261aaa9', '000000000000196786441b77252832c3': '0000000000001960b702799dee523428', '00000000000037daa2bcc5ed0da62496': '0000000000003394c782c6413af5c263'}


def compatible_source_ids(values):
    return {name: SOURCE_ID_COMPATIBILITY.get(value, value) for name, value in values.items()}


_FE_ONLY_KINDS = frozenset({
    "main_feg_single_run", "fixed_single_fe_endpoint", "fixed_single_fe_gravity_endpoint",
    "alpha10_feg_branch_medoid_continuation", "alpha10_branch_medoid_continuation",
    "final_fig6_vertical_compensated_fe", "triple_long_fixed_fe", "feg_main_alpha_run",
    "C3_gfe_warm_run", "C_GFE_single", "C_GFE_triple", "C_STD_commands",
})
_MIXED_SOLVER_KINDS = frozenset({
    "decision_endpoint_run", "final_single_vortex_run", "fig1_cartesian_domain_run",
    "multitrap_run", "morphology_solve", "revision_triple_ablation",
    "fig4_threepoint_selection_run", "final_alpha_run", "generality_cold_branch_run",
    "production_single_run", "production_triple_run", "appendix_single_run",
})
_AGGREGATE_KINDS = frozenset({"sota_benchmark"})


def _contains_fe(value):
    if isinstance(value, dict):
        if float(value.get("alpha_per_m", 0.0) or 0.0) > 0.0:
            return True
        return any(_contains_fe(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_fe(item) for item in value)
    return isinstance(value, str) and value.lower() in {
        "fe", "gfe", "fe+g", "force-equilibrium", "regularized-fe", "rh"}


def solver_cache_payload(kind, payload):
    if kind == "field_baseline_commands":
        # This IB/GS-only bank was tagged in the attached baseline notebook.
        # Keep that exact historical key; FE solver revisions do not affect it.
        return dict(payload, fe_solver_policy="fe-compact-lbfgsb-global-v1")
    affected = kind in _FE_ONLY_KINDS or kind in _AGGREGATE_KINDS or (
        kind in _MIXED_SOLVER_KINDS and _contains_fe(payload))
    if not affected:
        return payload
    return dict(payload, fe_solver_policy=FE_SOLVER_REVISION)

# Exact previously shipped namespaces; current payload and solver revision still apply.
COMPATIBLE_PRODUCTION_NAMESPACES = ('gfe-production-main-v2-0000000000000292',)
