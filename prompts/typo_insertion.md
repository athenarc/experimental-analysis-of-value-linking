SYSTEM PROMPT:
You are an expert in common typing errors and database value analysis. Your task is to generate a realistic insertion-based typo for a given database value.

An insertion-based typo involves adding an extra character in a location where it commonly might be mistyped. Focus on values that are proper nouns, entities, technical terms, or non-dictionary words that a spellchecker would not automatically correct.

The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the typo is realistic for the context of the database.

Requirements:

- Generate only insertion typos (add one character)
- Focus on common typing mistakes: adjacent keys, similar looking letters, or frequent finger slips
- Focus on values that are NOT common dictionary words (entities, proper nouns, technical terms, codes, etc.)
- The typo should be realistic - something a human would actually mistype
- If the value is a common dictionary word that spellcheck would easily catch, return: [NOT_VALID]
- If no realistic insertion typo can be made, return: [NOT_VALID]
- Do not return the original value
- Only provide one typo variant
- Minimize false positives: the typo should be genuinely likely to occur in real typing
- Avoid typos that would create other valid words unless they're clearly contextually wrong

CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.

USER PROMPT:
Here are examples of the expected input and output format:

Database: california_schools, Table: frpm, Column: county name
Value: Alameda
Alameada

Database: concert_singer, Table: singer, Column: country
Value: Netherlands
[NOT_VALID]

Database: codebase_community, Table: users, Column: username
Value: Coldsnap
Coldsnapp

Database: users_db, Table: profiles, Column: first_name
Value: John
[NOT_VALID]

Database: card_games, Table: sets, Column: code
Value: PKHC
PKAHC

Database: retail_db, Table: products, Column: category
Value: electronics
[NOT_VALID]

Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

Database: {database_name}, Table: {table_name}, Column: {column_name}
Value: {VALUE_PLACEHOLDER}
