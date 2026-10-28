import os
import unittest
from unittest.mock import patch
from arch_compare import configure, summarize, verify_arch


class ArchitectureComparisonTests(unittest.TestCase):
    def test_configuration_clears_inherited_overrides(self):
        with patch.dict(os.environ, PP_KERNEL="naive", PP_GRAPH="1", PP_ARCH="all", PP_FUTURE="1"):
            configure("r1")
            self.assertNotIn("PP_KERNEL", os.environ)
            self.assertNotIn("PP_FUTURE", os.environ)
            self.assertEqual(os.environ["PP_GRAPH"], "0")
            self.assertEqual(os.environ["PP_ARCH"], "r1")

    def test_dispatch_and_fallbacks(self):
        for radius, shape, kernel, sparse, expected in (
            (1,"NM","sparse_r1","1",("1","1","0")),
            (1,"NM","dense_bits","0",("0","0","0")),
            (12,"NN","tiled","0",("0","0","1")),
            (30,"NN","sparse_tiled","1",("1","0","1")),
            (31,"NN","tiled","0",("0","0","0")),
            (32,"NN","tiled","0",("0","0","0")),
            (5,"NM","sliding","0",("0","0","0")),
        ):
            record=dict(mode="all",merged=expected[0],r1_warp=expected[1],vn_diagonal=expected[2],r1_shared="0")
            row=dict(profile_kernel=kernel,sparse_selected=sparse)
            header=["128 128",f"R{radius},C4,M0,S1..2,B1..2,{shape}","dead","200"]
            verify_arch(record,row,header,"all")
            with self.assertRaises(ValueError):
                verify_arch(dict(record, mode="baseline"),row,header,"all")

    def test_shared_pair_dispatch(self):
        for kernel, selected in (("dense_r1","1"),("sparse_r1","1"),("dense_bits","0"),("tiled","0")):
            record=dict(mode="r1shared",merged="0",r1_warp="0",vn_diagonal="0",r1_shared=selected)
            row=dict(profile_kernel=kernel,sparse_selected=str(int(kernel.startswith('sparse'))))
            verify_arch(record,row,["128 128","R1,C4,M0,S2..3,B3..3,NM"],"r1shared")

    def test_pairing_outliers_and_wrong_outputs(self):
        samples=[dict(case_id="10",method=m,verdict="AC",compute_s=v)
                 for m,v in (("baseline","0.3"),("r1","0.1"),("r1","0.1"),("baseline","0.3"))]
        row=summarize(samples,("10",),"r1")[0]
        self.assertAlmostEqual(row["baseline_over_candidate"],3)
        self.assertEqual(row["timing_status"],"ok")
        samples[3]["compute_s"]="3.0"
        row=summarize(samples,("10",),"r1")[0]
        self.assertEqual(row["timing_status"],"unstable_samples")
        self.assertEqual(row["baseline_max_s"],3)
        samples[3]["verdict"]="WA"
        with self.assertRaises(ValueError): summarize(samples,("10",),"r1")
        with self.assertRaises(ValueError): summarize(samples[:3],("10",),"r1")


if __name__=="__main__": unittest.main()
