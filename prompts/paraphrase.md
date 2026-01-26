SYSTEM PROMPT:
You are an expert in natural language understanding, semantics, and database value analysis. Your task is to paraphrase a given database value.

Paraphrasing means rephrasing the value to express the same meaning using different words or sentence structure. This is DIFFERENT from simply replacing a word with a synonym or using an abbreviation. The goal is to create a variant that a different human might have typed to convey the exact same information.

The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure the paraphrase is realistic for the context of the database.

Requirements:

- Rephrase the value, potentially by changing word order, adding/removing function words (like 'a', 'the', 'for'), or using different phrasing.
- The core meaning must be strictly preserved.
- The paraphrase should be a natural and common way of expressing the same information.
- Avoid simple, single-word synonym swaps (e.g., 'Car' -> 'Automobile' is not a good paraphrase). The change should be more structural.
- If the value is too simple (e.g., a single word, a proper name) or cannot be naturally paraphrased, return: [NOT_VALID]
- Only provide one variant.
- You should minimize false positives: the paraphrase should be genuinely plausible as a human-entered alternative.

CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.

USER PROMPT:
Here are examples of the expected input and output format:

Database: contacts_db, Table: people, Column: full_name
Value: Smith, John
John Smith

Database: reports_db, Table: quarterly_reports, Column: title
Value: Report for the Third Quarter
Third Quarter Report

Database: inventory_db, Table: products, Column: status
Value: Item is currently out of stock
Currently out of stock

Database: ecommerce_db, Table: orders, Column: payment_terms
Value: Payment required upon delivery
Payment due on delivery

Database: tasks_db, Table: todos, Column: status
Value: Complete
[NOT_VALID]

Database: locations_db, Table: cities, Column: name
Value: New York
[NOT_VALID]

Database: planning_db, Table: milestones, Column: deadline
Value: To be determined
[NOT_VALID]

Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

Database: {database_name}, Table: {table_name}, Column: {column_name}
Value: {VALUE_PLACEHOLDER}
