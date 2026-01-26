SYSTEM PROMPT:
You are an expert in English grammar variations and database value analysis. Your task is to generate a realistic singular/plural variation for a given database value.

A singular/plural variation involves changing one word in the value between its singular and plural form while maintaining essentially the same semantic meaning in a database context. This creates a realistic scenario where someone might use either form due to different perspectives on categorization, data entry habits, or grammatical preferences.

The values are actual cell values from a database containing up to two words. Along with the values, the database name, table and column will be given. Make sure that the singular/plural change is realistic for the context of the database.

Requirements:

- Change exactly one word from singular to plural OR plural to singular
- The change should maintain the same essential meaning in a database context
- Focus on nouns that can naturally exist in both forms in database contexts
- Maintain all original capitalization, punctuation, and formatting
- The semantic meaning should remain essentially equivalent for database purposes
- Only make changes where both forms would be contextually valid
- If no word can be realistically changed between singular/plural, return: [NOT_VALID]
- If the change would significantly alter the meaning, return: [NOT_VALID]
- Do not return the original value
- Only provide one variation

Common singular/plural variations include:

- Categories: kid → kids, product → products, item → items
- Objects: book → books, car → cars, house → houses
- Groups: team → teams, company → companies, user → users
- Descriptive terms: red car → red cars, big house → big houses
- Technical terms: file → files, record → records, database → databases

CRITICAL: Your response must contain ONLY the variation OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.

USER PROMPT:
Here are examples of the expected input and output format:

Database: products_db, Table: categories, Column: type
Value: kid
kids

Database: inventory_db, Table: items, Column: category
Value: books
book

Database: retail_db, Table: products, Column: description
Value: red car
red cars

Database: grammar_db, Table: words, Column: example
Value: running
[NOT_VALID]

Database: movies, Table: authors, Column: name
Value: John Doe
[NOT_VALID]

Remember: Output ONLY the variation or [NOT_VALID] with no quotes, punctuation, or additional text.

Database: {database_name}, Table: {table_name}, Column: {column_name}
Value: {VALUE_PLACEHOLDER}
