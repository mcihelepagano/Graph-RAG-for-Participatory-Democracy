import pandas as pd

METAPOLICY_ASSIGNMENTS = {
    # 1 - Political Governance
    "We should adopt a multi-party system":         1,
    "We should oppose collectivism":                1,
    "We should introduce compulsory voting":        1,
    "We should adopt libertarianism":               1,

    # 2 - Criminal Justice & Law
    "We should abolish capital punishment":         2,
    "We should limit judicial activism":            2,
    "We should end racial profiling":               2,
    "Entrapment should be legalized":               2,
    "The use of public defenders should be mandatory": 2,
    "We should abolish the three-strikes laws":     2,
    "We should close Guantanamo Bay detention camp":2,
    "Holocaust denial should be a criminal offence":2,
    "We should ban targeted killing":               2,

    # 3 - Civil Liberties & Rights
    "We should prohibit flag burning":              3,
    "We should abolish safe spaces":                3,
    "We should abolish the right to keep and bear arms": 3,
    "We should cancel pride parades":               3,
    "We should prohibit women in combat":           3,
    "We should ban private military companies":     3,
    "We should fight for the abolition of nuclear weapons": 3,
    "Blockade of the Gaza Strip should be ended":   3,
    "We should end the use of economic sanctions":  3,

    # 4 - Bioethics & Personal Autonomy
    "Assisted suicide should be a criminal offence":4,
    "We should ban cosmetic surgery":               4,
    "We should ban cosmetic surgery for minors":    4,
    "We should legalize organ trade":               4,
    "We should legalize prostitution":              4,
    "We should subsidize embryonic stem cell research": 4,
    "We should ban human cloning":                  4,
    "We should legalize cannabis":                  4,
    "We should ban naturopathy":                    4,
    "Homeopathy brings more harm than good":        4,
    "We should legalize sex selection":             4,
    "Surrogacy should be banned":                   4,

    # 5 - Economic & Labor Policy
    "We should subsidize student loans":            5,
    "We should limit executive compensation":       5,
    "We should subsidize stay-at-home dads":        5,
    "We should end mandatory retirement":           5,
    "We should subsidize vocational education":     5,
    "We should adopt an austerity regime":          5,
    "We should subsidize journalism":               5,
    "We should subsidize space exploration":        5,
    "Payday loans should be banned":                5,
    "We should ban algorithmic trading":            5,

    # 6 - Religion & Secularism
    "We should ban the Church of Scientology":      6,
    "We should prohibit school prayer":             6,
    "We should adopt atheism":                      6,
    "We should ban missionary work":                6,
    "The vow of celibacy should be abandoned":      6,

    # 7 - Gender, Family & Social Norms
    "We should abandon marriage":                   7,
    "We should legalize polygamy":                  7,
    "We should adopt gender-neutral language":      7,
    "We should end affirmative action":             7,
    "We should legalize sex selection":             7,

    # 8 - Education, Media & Technology
    "Homeschooling should be banned":               8,
    "We should ban the use of child actors":        8,
    "We should adopt a zero-tolerance policy in schools": 8,
    "Intelligence tests bring more harm than good": 8,
    "We should abandon the use of school uniform":  8,
    "We should stop the development of autonomous cars": 8,
    "Social media brings more harm than good":      8,
    "We should subsidize Wikipedia":                8,
    "We should abandon television":                 8,
    "We should ban telemarketing":                  8,

    # 9 - Animal Welfare & Environment
    "We should ban whaling":                        9,
    "We should abolish zoos":                       9,
    "We should ban fast food":                      9,
    "We should ban factory farming":                9,
    "We should fight urbanization":                 9,
    "We should abolish the Olympic Games":          9,

    # 10 - Child & Family Welfare
    "Foster care brings more harm than good":       10,
    "We should abolish intellectual property rights": 5,
}

METAPOLICY_NAMES = {
    1:  "Political Governance",
    2:  "Criminal Justice & Law",
    3:  "Civil Liberties & Rights",
    4:  "Bioethics & Personal Autonomy",
    5:  "Economic & Labor Policy",
    6:  "Religion & Secularism",
    7:  "Gender, Family & Social Norms",
    8:  "Education, Media & Technology",
    9:  "Animal Welfare & Environment",
    10: "Child & Family Welfare"
}

# --- Apply ---
policy_df = pd.read_csv('policies.csv')
policy_df['metapolicy_id'] = policy_df['policy'].map(METAPOLICY_ASSIGNMENTS)
policy_df['metapolicy_name'] = policy_df['metapolicy_id'].map(METAPOLICY_NAMES)

# --- Verify no unmapped policies ---
unmapped = policy_df[policy_df['metapolicy_id'].isna()]
if len(unmapped) > 0:
    print("WARNING - Unmapped policies:")
    print(unmapped['policy'].tolist())
else:
    print("All policies mapped successfully")

# --- Summary ---
print("\nMetaPolicy distribution:")
summary = policy_df.groupby(['metapolicy_id', 'metapolicy_name'])\
                   .size().reset_index(name='n_policies')
print(summary.to_string())

policy_df.to_csv('policies.csv', index=False)
print("\nSaved policies.csv")