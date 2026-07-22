from painconnect_rd.pipeline import build_stub_profiles, public_source_catalog


def test_public_source_catalog_mentions_monarch():
    sources = public_source_catalog()
    assert any(source.name == "Monarch / DisMech" for source in sources)


def test_stub_profiles_include_core_scn9a_diseases():
    profiles = build_stub_profiles()
    names = {profile.disease for profile in profiles}
    assert "Primary erythromelalgia" in names
    assert "Paroxysmal extreme pain disorder" in names
    assert "Congenital insensitivity to pain" in names
