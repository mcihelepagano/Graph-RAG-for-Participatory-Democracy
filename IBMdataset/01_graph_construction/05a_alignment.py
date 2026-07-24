import pandas as pd

policy_df = pd.read_csv('policies.csv')

# Reassign Foster care to MetaPolicy 7
policy_df.loc[
    policy_df['policy'] == 'Foster care brings more harm than good',
    ['metapolicy_id', 'metapolicy_name']
] = [7, 'Gender, Family & Social Norms']

# Drop MetaPolicy 10 — no longer needed
# Verify
print("Updated MetaPolicy distribution:")
print(policy_df.groupby(['metapolicy_id', 'metapolicy_name'])
      .size().reset_index(name='n_policies').to_string(index=False))

unmapped = policy_df[policy_df['metapolicy_id'].isna()]
print(f"\nUnmapped: {len(unmapped)}")

policy_df.to_csv('policies.csv', index=False)
print("\nSaved policies.csv")