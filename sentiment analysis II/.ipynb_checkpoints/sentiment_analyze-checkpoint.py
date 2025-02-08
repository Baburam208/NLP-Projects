import pandas as pd
from ftfy import fix_text

# Sample DataFrame
df = pd.DataFrame({'text': [
    "whenever i go to a place that doesnâ€™t take apple pay",
    "à´•àµ‡à´°à´³à´¤àµà´¤à´¿àµ½ à´†à´¦àµà´¯à´®à´¾à´¯à´¿",
    "This is a normal sentence."
]})

# Fix encoding issues
df['text_fixed'] = df['text'].apply(fix_text)

# Show results
print(df[['text', 'text_fixed']])
