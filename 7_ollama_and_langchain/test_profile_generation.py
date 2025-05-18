from profile_util import tabular_frame, generate_profile_report

# Get the DataFrame
df = tabular_frame(
    industry='finance',
    num_rows=10
)

# Generate the data profile
generate_profile_report(
    df=df,
    file_output_path='result.html',
    explorative=True
)
