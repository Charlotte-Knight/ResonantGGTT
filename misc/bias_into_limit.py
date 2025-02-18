import json
import optimisation.limit
import numpy as np

# N = 2
# bias = 1.0
# res = {
#         "category_boundaries": [
#             0,
#             0.9557744264602661,
#             0.9790858030319214,
#             0.9867178201675415,
#             0.9904778003692627,
#             0.9931322336196899,
#             0.995099663734436,
#             0.997249960899353,
#             0.9984714984893799,
#             0.9990220069885254,
#             0.9991075992584229,
#             0.9994710683822632,
#             0.9996082186698914,
#             0.999616801738739,
#             0.99967360496521,
#             0.9998748302459717,
#             1
#         ],
#         "nbkgs": [
#             1803.8277978662363,
#             23.468080525190413,
#             5.024427942619781,
#             2.3540330983244027,
#             2.519836050354805,
#             1.2162833665912272,
#             1.1119387638976685,
#             0.5565949982817443,
#             0.10150732835829715,
#             0.13453658996023116,
#             0.13595199392537355,
#             0.10097302666711565,
#             0.1416488284843425,
#             0.07277967941829618,
#             0.07219116483998583,
#             0.05811640666352912
#         ],
#         "nsigs": [
#             0.17231394350528717,
#             0.19668380916118622,
#             0.16817818582057953,
#             0.15763235092163086,
#             0.24926728010177612,
#             0.2712634801864624,
#             0.8328419923782349,
#             1.022566318511963,
#             1.1464484930038452,
#             0.23579874634742737,
#             1.343658685684204,
#             0.6938992738723755,
#             0.32662028074264526,
#             0.7019529342651367,
#             2.9017364978790283,
#             11.909623146057129
#         ],
#         "optimal_limit": 0.14944076538085938,
#         "score": "intermediate_transformed_score_NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90",
#         "sig_proc": "NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90"
#     }

# N=5
# bias = 0.5
# res = {
#         "category_boundaries": [
#             0,
#             0.9658257961273193,
#             0.9824720621109009,
#             0.9871355295181274,
#             0.9915047883987427,
#             0.9931972026824951,
#             0.9957452416419983,
#             0.9972098469734192,
#             0.9977794289588928,
#             0.9988917112350464,
#             0.9994710683822632,
#             0.9996297359466553,
#             1
#         ],
#         "nbkgs": [
#             1771.376280188834,
#             13.191261517755084,
#             3.091722192844339,
#             3.066699729169703,
#             1.5274352646496634,
#             1.483382935146734,
#             0.5412769250705032,
#             0.3683933649534508,
#             0.2944949138222977,
#             0.3434072811961271,
#             0.34030183403477987,
#             0.14494148123050304
#         ],
#         "nsigs": [
#             0.23395410180091858,
#             0.2020147740840912,
#             0.11860811710357666,
#             0.23304790258407593,
#             0.16431836783885956,
#             0.4231227934360504,
#             0.6515383124351501,
#             0.27473264932632446,
#             1.6050418615341187,
#             1.8908532857894897,
#             1.0901674032211304,
#             15.443008422851562
#         ],
#         "optimal_limit": 0.14905929565429688,
#         "score": "intermediate_transformed_score_NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90",
#         "sig_proc": "NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90"
#    }

N=10
bias = 0.4
res = {
        "category_boundaries": [
            0,
            0.9630184173583984,
            0.9785513281822205,
            0.9877123832702637,
            0.99173903465271,
            0.9937925934791565,
            0.995099663734436,
            0.9964394569396973,
            0.9977794289588928,
            0.9994710683822632,
            1
        ],
        "nbkgs": [
            1781.0699940089523,
            14.470444720863748,
            6.081431476260814,
            3.1579462779546668,
            1.4728986273984173,
            0.7464494702077968,
            0.7199825847839031,
            0.7350281821159335,
            0.662682259541019,
            0.6528528151395973
        ],
        "nsigs": [
            0.21303784847259521,
            0.14935201406478882,
            0.2164279967546463,
            0.22274592518806458,
            0.2151123583316803,
            0.19909849762916565,
            0.4283812344074249,
            0.658196747303009,
            3.4956603050231934,
            16.531461715698242
        ],
        "optimal_limit": 0.18434524536132812,
        "score": "intermediate_transformed_score_NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90",
        "sig_proc": "NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90"
    }

# N=20
# bias = 0.2
# res = {
#         "category_boundaries": [
#             0,
#             0.9656693935394287,
#             0.9817337989807129,
#             0.9867178201675415,
#             0.9913000464439392,
#             0.9926824569702148,
#             0.995099663734436,
#             0.9977794289588928,
#             1
#         ],
#         "nbkgs": [
#             1789.495450706292,
#             13.32728335070707,
#             3.132110234721934,
#             2.942624473087376,
#             1.7003692054351258,
#             1.4693575555365814,
#             1.4529968394315829,
#             1.3047009154167417
#         ],
#         "nsigs": [
#             0.22935307025909424,
#             0.18919089436531067,
#             0.11700184643268585,
#             0.21798351407051086,
#             0.1473749577999115,
#             0.3133293092250824,
#             1.0863384008407593,
#             20.02576446533203
#         ],
#         "optimal_limit": 0.19359588623046875,
#         "score": "intermediate_transformed_score_NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90",
#         "sig_proc": "NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90"
#     }


s = np.array(res["nsigs"])
b = np.array(res["nbkgs"])
N_sidebands = (b/0.65)*10
print(N_sidebands)
b_err = 1/np.sqrt(N_sidebands) * b
print(b_err)

# print("".ljust(20) + "Limit".ljust(10) + "Relative diff (%)")
# nom = optimisation.limit.calculateExpectedLimit(s, b, np.zeros_like(b))
# print("Nominal limit:".ljust(20) + ("%.4f"%nom).ljust(10))

# bp = optimisation.limit.calculateExpectedLimit(s, b+bias*b_err, np.zeros_like(b), rhigh=1000)
# print(f"B += {bias}*b_err".ljust(20) + ("%.4f"%bp).ljust(10) + ("%.4f"%(100*(bp-nom)/nom)))

# bp = optimisation.limit.calculateExpectedLimit(s, np.clip(b-bias*b_err, 0.0001, a_max=None), np.zeros_like(b), rhigh=1000)
# print(f"B -= {bias}*b_err".ljust(20) + ("%.4f"%bp).ljust(10) + ("%.4f"%(100*(bp-nom)/nom)))

# bp = optimisation.limit.calculateExpectedLimit(s+bias*b_err, b, np.zeros_like(b), rhigh=1000)
# print(f"S += {bias}*b_err".ljust(20) + ("%.4f"%bp).ljust(10) + ("%.4f"%(100*(bp-nom)/nom)))

# bp = optimisation.limit.calculateExpectedLimit(s-bias*b_err, b, np.zeros_like(b), rhigh=1000)
# print(f"S -= {bias}*b_err".ljust(20) + ("%.4f"%bp).ljust(10) + ("%.4f"%(100*(bp-nom)/nom)))

print("".ljust(20) + "Limit".ljust(10) + "Relative diff (%)")
nom = optimisation.limit.calculateObservedLimit(s, b, b)
print("Nominal limit:".ljust(20) + ("%.4f"%nom).ljust(10))

bp = optimisation.limit.calculateObservedLimit(s, b, b+bias*b_err, rhigh=1000)
print(f"B += {bias}*b_err".ljust(20) + ("%.4f"%bp).ljust(10) + ("%.4f"%(100*(bp-nom)/nom)))

bp = optimisation.limit.calculateObservedLimit(s, b, np.clip(b-bias*b_err, 0.0001, a_max=None), rhigh=1000)
print(f"B -= {bias}*b_err".ljust(20) + ("%.4f"%bp).ljust(10) + ("%.4f"%(100*(bp-nom)/nom)))