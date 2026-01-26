SYSTEM PROMPT:
You are an expert in semantics, logic, and linguistic transformations. Your task is to generate a specific type of paraphrase for a given database value: a negated antonym.

This transformation involves replacing a value with the negation of its opposite (its antonym). The transformation follows the pattern: `[value]` -> `not [antonym of value]`. This creates a phrase that is semantically equivalent to the original value. For example, the value 'active' would be transformed into 'not inactive'.

This is DIFFERENT from a simple antonym ('active' -> 'inactive') or a simple negation ('active' -> 'not active'). The meaning must be preserved.

The values are actual cell values from a database, so the database name, table, and column will be given. The transformation should only be applied if it results in a natural-sounding phrase. This is most common for adjectives and status words.

Requirements:

- Replace the value with 'not' followed by its direct, common antonym.
- The resulting phrase must be semantically equivalent to the original value.
- The antonym used must be a common and natural opposite (e.g., the antonym of 'possible' is 'impossible').
- If the value has no clear, common antonym, or if the transformation would be awkward, return: [NOT_VALID]
- If the value is a complex phrase, a proper name, or is already in a negative form, return: [NOT_VALID]
- Only provide one variant.

CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.

USER PROMPT:
Here are examples of the expected input and output format:

Database: user_accounts, Table: users, Column: status
Value: active
not inactive

Database: forms_db, Table: fields, Column: validation
Value: required
not optional

Database: compliance_db, Table: actions, Column: status
Value: legal
not illegal

Database: security_db, Table: permissions, Column: access_level
Value: allowed
not forbidden

Database: contacts_db, Table: people, Column: name
Value: John Smith
[NOT_VALID]

Database: inventory_db, Table: products, Column: color
Value: blue
[NOT_VALID]

Database: reports_db, Table: quarterly_reports, Column: title
Value: Report for the Third Quarter
[NOT_VALID]

Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

Database: {database_name}, Table: {table_name}, Column: {column_name}
Value: {VALUE_PLACEHOLDER}
