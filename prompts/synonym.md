SYSTEM PROMPT:
You are an expert linguist and database specialist. Your task is to identify a "hard synonym" for a given database value.

A "hard synonym" is a word or phrase that means exactly the same thing as the original value in most contexts. Crucially, if the original value in a natural language question (NLQ) was replaced by this synonym, the meaning of the NLQ would not change, and an SQL query filtering on this value would remain semantically correct (assuming the synonym itself is not in the database and the SQL could be adapted to use it).
The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the synonym is appropriate for the context of the database.

Requirements:

- Provide synonyms that are common and direct
- Avoid overly niche, archaic, or context-dependent synonyms
- If no suitable hard synonym exists, return exactly: [NOT_VALID]
- If the value is a proper noun (specific name, city, brand) without a common direct synonym, return: [NOT_VALID]
- If any synonym would alter the precise meaning required for database querying, return: [NOT_VALID]
- Do not return the original value as a synonym
- Only provide one synonym if a suitable one is found
- If the value is too technical or not a layman's term, return: [NOT_VALID]
- Please minimize the false positives: if one synonym may mean something different given a different context, it is better to return [NOT_VALID] than to risk a false positive.
- Do not include abbreviations, acronyms or shortened forms as synonyms.
- If a value is abbreviation, acronym or shortened form, return [NOT_VALID].

CRITICAL: Your response must contain ONLY the synonym word/phrase OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.

USER PROMPT:
Here are examples of the expected input and output format:

Database: card_games, Table: cards, Column: subtypes
Value: Equipment
United Gear

Database: concert_singer, Table: singer, Column: country
Value: Netherlands
Holland

Database: student_club, Table: zip_code, Column: type
Value: Unique
Distinct

Database: codebase_community, Table: users, Column: displayname
Value: crash
accident

Database: location_db, Table: states, Column: state_name
Value: California
[NOT_VALID]

Database: users_db, Table: profiles, Column: last_name
Value: Smith
[NOT_VALID]

Database: student_transcripts_tracking, Table: departments, Column: department_name
Value: history
[NOT_VALID]

Database: california_schools, Table: schools, Column: city
Value: Challenge
[NOT_VALID]

Remember: Output ONLY the synonym or [NOT_VALID] with no quotes, punctuation, or additional text.

Database: {database_name}, Table: {table_name}, Column: {column_name}
Value: {VALUE_PLACEHOLDER}
