"""Unit tests for src/preprocessing (normalize, translit_vocab) and src/io/data_loader.py chunked helpers.

Run from the project root (no extra packages needed - stdlib unittest):
    python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from src.preprocessing.normalize import (OUTPUT_FIELDS, Normalizer, dominant_script,  # noqa: E402
                                         fix_leet, native_key, normalize_frame, skeleton,
                                         unicode_canonical)

NZ = Normalizer(vocab_path="")          # rules only, no learned vocab


class TestUnicode(unittest.TestCase):
    def test_nfkc_and_zero_width(self):
        self.assertEqual(unicode_canonical("Ａｃｍｅ​  Corp"), "Acme Corp")

    def test_none_and_nan(self):
        self.assertEqual(unicode_canonical(None), "")
        self.assertEqual(unicode_canonical(float("nan")), "")

    def test_non_ascii_is_kept_in_canonical_form(self):
        # canonicalisation must NOT delete non-ASCII text
        self.assertEqual(unicode_canonical("श्री मीडिया"), "श्री मीडिया")
        self.assertEqual(unicode_canonical("Général"), "Général")

    def test_dominant_script(self):
        self.assertEqual(dominant_script("श्री मीडिया Pvt"), "DEVANAGARI")
        self.assertEqual(dominant_script("Café Général"), "LATIN")
        self.assertEqual(dominant_script("1234"), "NONE")
        self.assertEqual(dominant_script("சன்ரைஸ்"), "TAMIL")

    def test_native_key_strips_punctuation(self):
        self.assertEqual(native_key("प्रा."), "प्रा")


class TestTokenHelpers(unittest.TestCase):
    def test_leet(self):
        self.assertEqual(fix_leet("6eneral"), "general")
        self.assertEqual(fix_leet("we1lness"), "wellness")
        self.assertEqual(fix_leet("cypre5s"), "cypress")
        self.assertEqual(fix_leet("5w"), "sw")

    def test_leet_leaves_numbers_and_ordinals(self):
        for t in ["2nd", "4th", "22400", "b2b", "12345a"]:
            self.assertEqual(fix_leet(t), t)

    def test_skeleton_matches_transliterations(self):
        self.assertEqual(skeleton("private"), skeleton("praivet"))
        self.assertEqual(skeleton("limited"), skeleton("limtid"))
        self.assertEqual(skeleton("maharashtra"), skeleton("mharastr"))
        self.assertEqual(skeleton("fillmore"), skeleton("fillmmore"))
        self.assertEqual(skeleton("123"), "123")


class TestNames(unittest.TestCase):
    def n(self, s):
        return NZ.name(s)

    def test_legal_forms_removed_from_core_and_collected(self):
        r = self.n("Raka Motors Private Limited")
        self.assertEqual(r["name_core"], "raka motors")
        self.assertEqual(r["name_legal"], "ltd pvt")

    def test_shuffled_legal_forms_give_same_core(self):
        self.assertEqual(self.n("-- Raka Limited Private Motors")["name_core"],
                         self.n("Raka Motors Pvt. Ltd.")["name_core"])

    def test_dotted_legal_forms(self):
        self.assertEqual(self.n("PRIME INVESTMENTS L.L.P.")["name_legal"], "llp")
        self.assertEqual(self.n("Vetasoft Carolina P.C.")["name_core"], "vetasoft carolina")

    def test_french_legal_forms(self):
        for s, legal in [("Oi Amis SARL", "sarl"), ("Ets Comlunale Groupe SASU", "sasu"),
                         ("Ets Deleves EURL", "eurl"), ("Passion Parents SA", "sa")]:
            self.assertEqual(self.n(s)["name_legal"], legal)

    def test_trade_name_markers(self):
        for s in ["Nexnex t/a Raka Motors Private Limited",
                  "Gildiri d/b/a Raka Motors Private Limited",
                  "Pyrariza DBA: Raka Motors Private Limited",
                  "Wexfaye aka Raka Motors Private Limited",
                  "Umbrajaxhalo née Raka Motors Private Limited"]:
            r = self.n(s)
            self.assertEqual(r["name_core"], "raka motors", s)
            self.assertTrue(r["name_alt"], s)

    def test_accents_case_punctuation(self):
        self.assertEqual(self.n("Oncology Care Associates of West Linn Có")["name_core"],
                         self.n("ONCOLOGY CARE ASSOCIATES OF WEST LINN")["name_core"])

    def test_ampersand_and(self):
        self.assertEqual(self.n("Raj & Sons")["name_core"], self.n("Raj and Sons")["name_core"])

    def test_url_suffix_and_trailing_id(self):
        self.assertEqual(self.n("Atlantic Cómmittee | www.atlanticc.com")["name_core"], "atlantic committee")
        self.assertEqual(self.n("Golden Cónsúltants Limited - 1431181809")["name_core"], "golden consultants")

    def test_domain_only_name_is_kept(self):
        r = self.n("deepeshagri.com")
        self.assertEqual(r["name_core"], "deepeshagri")
        self.assertEqual(self.n("Deepesh Agri Pvt Ltd")["name_compact"], "deepeshagri")

    def test_honorifics_and_prefix_junk(self):
        self.assertEqual(self.n("Mr HUGE SOLUTIONS PRIVATE")["name_core"], "huge solutions")
        self.assertEqual(self.n("Smt Prime Investments")["name_core"], "prime investments")
        self.assertEqual(self.n("M/s Prime Investments")["name_core"], "prime investments")
        self.assertEqual(self.n(">> @Prime Investments")["name_core"], "prime investments")

    def test_repeated_token(self):
        self.assertEqual(self.n("NQ NQ Anchor Freedom LLC")["name_core"], "nq anchor freedom")

    def test_native_script_is_transliterated_not_deleted(self):
        r = self.n("श्री मीडिया प्राइवेट लिमिटेड")
        self.assertEqual(r["name_translit"], 1)
        self.assertEqual(r["name_script"], "DEVANAGARI")
        self.assertIn("midiya", r["name_core"])
        self.assertIn("ltd", r["name_legal"])

    def test_learned_vocab_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "v.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"vocab": {"प्राइवेट": "private", "मीडिया": "media", "श्री": "sree"}}, f)
            nz = Normalizer(vocab_path=p)
            r = nz.name("श्री मीडिया प्राइवेट लिमिटेड")
            self.assertEqual(r["name_core"], "sree media")
            self.assertEqual(r["name_legal"], "ltd pvt")

    def test_only_legal_words_falls_back(self):
        self.assertTrue(self.n("Private Limited")["name_core"])

    def test_empty(self):
        r = self.n("")
        self.assertEqual(r["name_core"], "")
        self.assertEqual(r["name_script"], "NONE")


class TestAddresses(unittest.TestCase):
    def a(self, s, c):
        return NZ.address(s, c)

    def test_abbreviations_canonical(self):
        x = self.a("5548 MT TABOR RD, GREENWOOD, FL", "US")
        y = self.a("5548 Mount Tabor Road, Greenwood, Florida", "US")
        self.assertEqual(x["addr_core"], y["addr_core"])
        self.assertEqual(x["addr_state"], "US-FL")
        self.assertEqual(y["addr_state"], "US-FL")

    def test_saint_street_confusion(self):
        self.assertEqual(self.a("145 CHERRYWOOD SAINT, CA", "US")["addr_core"],
                         self.a("145 Cherrywood St, California", "US")["addr_core"])

    def test_numbers(self):
        self.assertEqual(self.a("#00111 Newton Street", "US")["addr_nums"], "111")
        self.assertEqual(self.a("242-244 Main St", "US")["addr_nums"], "242 244")
        self.assertEqual(self.a("613 Parek Market39 J S S Road", "India")["addr_nums"], "39 613")

    def test_placeholders(self):
        self.assertEqual(self.a("<NULL>", "US")["addr_missing"], 1)
        self.assertEqual(self.a("N/A", "US")["addr_missing"], 1)
        self.assertEqual(self.a("", "US")["addr_missing"], 1)
        r = self.a("1318 1/2 Division St, <NULL>, City Of Appleton, Wisconsin", "US")
        self.assertNotIn("null", r["addr_clean"])
        self.assertEqual(r["addr_state"], "US-WI")

    def test_india_state_codes_and_names(self):
        self.assertEqual(self.a("Satara, MH", "India")["addr_state"], "IN-MH")
        self.assertEqual(self.a("Satara, Maharashtra", "India")["addr_state"], "IN-MH")
        self.assertEqual(self.a("Mahad, Raigarh(Mh)", "India")["addr_state"], "IN-MH")
        self.assertEqual(self.a("Jaipur, राजस्थान", "India")["addr_state"], "IN-RJ")
        self.assertEqual(self.a("Uppal, తెలంగాణ", "India")["addr_state"], "IN-TS")

    def test_france_region_and_department_agree(self):
        x = self.a("13 RUE DU 14 JUILLET, LA TESTE-DE-BUCH, Gironde", "France")
        y = self.a("13 Rue du 14 Juillet, La Teste-de-Buch, Nouvelle-Aquitaine", "France")
        self.assertEqual(x["addr_state"], "FR-NAQ")
        self.assertEqual(y["addr_state"], "FR-NAQ")
        self.assertEqual(x["addr_core"], y["addr_core"])

    def test_french_street_abbreviations(self):
        self.assertEqual(self.a("278 R. WINSTON CHURCHILL, Dunkerque", "France")["addr_core"],
                         self.a("278 Rue Winston Churchill, Dunkerque", "France")["addr_core"])

    def test_department_word_inside_street_is_not_a_state(self):
        r = self.a("81 Cours De La Somme, Bordeaux", "France")
        self.assertIn("somme", r["addr_core"])

    def test_unknown_country_is_open_set(self):
        r = self.a("12 Main Street, Springfield, Ohio", "Canada")   # unseen label: no crash
        self.assertEqual(r["addr_state"], "US-OH")
        self.assertEqual(self.a("1 Some Road", "")["addr_nums"], "1")


class TestFrame(unittest.TestCase):
    def test_originals_preserved_and_fields_added(self):
        import pandas as pd
        df = pd.DataFrame({"entity_id": ["S2-1", "S2-2"],
                           "business_name": ["श्री मीडिया प्रा. लि.", "Café Général SARL"],
                           "business_address": ["Pune, महाराष्ट्र", "N/A"],
                           "country": ["India", "France"]})
        before = df.copy()
        out = normalize_frame(df, NZ)
        pd.testing.assert_frame_equal(df, before)                     # input untouched
        for c in before.columns:
            self.assertTrue((out[c] == before[c]).all())              # originals kept as-is
        for c in OUTPUT_FIELDS:
            self.assertIn(c, out.columns)
        self.assertEqual(out.loc[1, "addr_missing"], 1)


class TestVocabLearner(unittest.TestCase):
    def test_cooccurrence_beats_frequent_lookalike(self):
        from src.preprocessing.translit_vocab import VocabLearner
        L = VocabLearner()
        for i in range(10):
            # 'care' co-occurs with the Tamil word for 'future' only 60% of the time
            L.add_pair("Future Care Private Limited" if i < 6 else "Future Tech Private Limited",
                       "ஃப்யூச்சர் கேர் பிரைவேட் லிமிடெட்")
            L.add_pair("Green Care Clinic" if i < 6 else "Green Foods", "கிரீன் கேர்")
            L.add_pair("Kochi City Traders", "കൊച്ചി സിറ്റി ട്രേഡേഴ്സ്")
            L.add_pair("Sree Foundation" if i < 5 else "Modern Foundation", "శ్రీ ఫౌండేషన్")
            L.add_pair("Chennai, Tamil Nadu", "சென்னை, தமிழ்நாடு")
        v = L.build(min_count=3)
        self.assertEqual(v["ஃப்யூச்சர்"], "future")
        self.assertEqual(v["கேர்"], "care")
        self.assertEqual(v["கிரீன்"], "green")
        self.assertEqual(v["സിറ്റി"], "city")
        self.assertEqual(v["കൊച്ചി"], "kochi")
        self.assertEqual(v["ఫౌండేషన్"], "foundation")
        self.assertEqual(v["தமிழ்நாடு"], "tamil nadu")
        self.assertEqual(v["லிமிடெட்"], "limited")


class TestKeysAndIndexes(unittest.TestCase):
    def test_stable_hash_is_deterministic(self):
        import subprocess
        from src.preprocessing.keys import stable_hash
        code = "from src.preprocessing.keys import stable_hash;print(int(stable_hash(['US#acme'])[0]))"
        other = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(int(other.stdout.strip()), int(stable_hash(["US#acme"])[0]))

    def test_index_round_trip_and_country_isolation(self):
        import pandas as pd
        from src.preprocessing.build_indexes import build, candidates
        from src.preprocessing.preprocess import normalize_df_parallel
        rows = {
            1: [("S1-1", "Raka Motors Pvt Ltd", "74 Kesarkar Peth, Satara, MH", "India"),
                ("S1-2", "Acme Corp", "12 Main Street, Dallas, TX", "US")],
            2: [("S2-1", "RAKA MOTORS PRIVATE LIMITED", "74 KESARKAR PETH, SATARA", "India"),
                ("S2-2", "Raka Motors", "74 Kesarkar Peth", "US")],          # other country
            3: [("S3-1", "Acme Corporation", "12 Main St, Dallas, Texas", "US")],
        }
        with tempfile.TemporaryDirectory() as d:
            for k, rs in rows.items():
                df = pd.DataFrame(rs, columns=["entity_id", "business_name", "business_address", "country"])
                os.makedirs(os.path.join(d, "train"), exist_ok=True)
                normalize_df_parallel(df, workers=1, vocab_path="").to_parquet(
                    os.path.join(d, "train", f"source{k}.parquet"), index=False)
            stats = build(d, ("train",), ["name_first", "num_street"], max_block=10)
            self.assertIn("name_first", stats["train"])
            c = candidates(d, "train", ["name_first", "num_street"])
            pairs = set(zip(c.s1_entity_id, c.other_entity_id))
            self.assertIn(("S1-1", "S2-1"), pairs)
            self.assertIn(("S1-2", "S3-1"), pairs)
            self.assertNotIn(("S1-1", "S2-2"), pairs)                        # never across countries
            n = dict(zip(zip(c.s1_entity_id, c.other_entity_id), c.n_keys))
            self.assertEqual(n[("S1-1", "S2-1")], 2)                         # shares both keys


class TestLoader(unittest.TestCase):
    def test_quotes_and_empty_fields_survive(self):
        from src.io.data_loader import iter_tsv, read_tsv
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.tsv")
            with open(p, "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write('S2-1\tDominica"s "Brewing\t\tUS\n')
                f.write("S2-2\tNA\tnull\tIndia\n")
            df = read_tsv(p)
            self.assertEqual(len(df), 2)
            self.assertEqual(df.loc[0, "business_name"], 'Dominica"s "Brewing')
            self.assertEqual(df.loc[0, "business_address"], "")        # empty, not NaN
            self.assertEqual(df.loc[1, "business_name"], "NA")          # not converted to NaN
            self.assertEqual(sum(len(c) for c in iter_tsv(p, chunksize=1)), 2)


if __name__ == "__main__":
    unittest.main()
