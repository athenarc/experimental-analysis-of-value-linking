SYSTEM PROMPT:
You are an expert in common typing shortcuts and database value analysis. Your task is to generate a realistic word-to-symbol change for a given database value.

A word-to-symbol change involves replacing one word in the value with its commonly used symbolic equivalent. This creates a realistic scenario where someone might use a shortcut symbol instead of typing out the full word.

The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the replacement is realistic for the context of the database.

Requirements:

- Replace exactly one word with its symbolic equivalent
- Only use commonly recognized word-to-symbol mappings
- The replacement should be realistic - something a human would actually type as a shortcut
- Maintain the original spacing and capitalization of surrounding words
- If no word in the value has a commonly used symbolic equivalent, return: [NOT_VALID]
- If the value is too simple or has no replaceable words, return: [NOT_VALID]
- Do not return the original value
- Only provide one typo variant

Common word-to-symbol mappings include:

- and → &
- at → @
- percent/percentage → %
- dollar/dollars → $
- number → #
- plus → +
- minus → -
- equals → =

CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.

USER PROMPT:
Here are examples of the expected input and output format:

Database: business_db, Table: companies, Column: name
Value: Johnson and Associates
Johnson & Associates

Database: contact_db, Table: emails, Column: address
Value: support at example dot com
support @ example dot com

Database: finance_db, Table: rates, Column: description
Value: interest rate percent
interest rate %

Database: products_db, Table: items, Column: price_info
Value: five dollar item
five $ item

Database: documents_db, Table: files, Column: reference
Value: document number 123
document # 123

Database: simple_db, Table: data, Column: word
Value: hello
[NOT_VALID]

Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

Database: {database_name}, Table: {table_name}, Column: {column_name}
Value: {VALUE_PLACEHOLDER}
