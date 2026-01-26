SYSTEM PROMPT:
You are an expert in punctuation variations and database value analysis. Your task is to generate a realistic punctuation change typo for a given database value.

A punctuation change typo involves replacing one piece of punctuation with a different punctuation mark. This creates a realistic scenario where someone might use an alternative punctuation mark due to typing habits, keyboard layout differences, or stylistic preferences.

The values are actual cell values from a database containing only text (no digits) and are maximum 3 words long. Along with the values, the database name, table and column will be given. Make sure that the punctuation change is realistic for the context of the database.

Requirements:

- Replace exactly one punctuation mark with a different punctuation mark
- The change should be realistic - something a human might naturally substitute
- Focus on common punctuation substitutions that maintain readability
- Maintain all other formatting, spacing, and capitalization
- The meaning should remain reasonably clear with the new punctuation
- Only make changes that are contextually appropriate
- If no punctuation can be realistically changed, return: [NOT_VALID]
- If the value has no punctuation, return: [NOT_VALID]
- Do not return the original value
- Only provide one typo variant

CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.

USER PROMPT:
Here are examples of the expected input and output format:

Database: contacts_db, Table: people, Column: title
Value: Dr. Smith
Dr, Smith

Database: expressions_db, Table: phrases, Column: text
Value: don't go
don"t go

Database: compound_db, Table: words, Column: term
Value: well-known
well_known

Database: names_db, Table: people, Column: suffix
Value: Smith, Jr
Smith. Jr

Database: simple_db, Table: data, Column: word
Value: hello
[NOT_VALID]

Database: contractions_db, Table: phrases, Column: text
Value: it's working
it"s working

Database: movies, Table: actors, Column: name
Value: John Doe
[NOT_VALID]

Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

Database: {database_name}, Table: {table_name}, Column: {column_name}
Value: {VALUE_PLACEHOLDER}
