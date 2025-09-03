
class SFTTemplate:
    def __init__(self, inst: str, output: str, input_fields, output_fields, template_id, first_turn=None):
        self.first_turn = first_turn
        self.input_template = inst
        self.output_template = output
        self.template_id = template_id
        self.task = template_id.split('-')[0]

        # --- MODIFICATION START ---
        # We make the assertion more generic to allow for new tasks.
        # This checks that the provided fields are strings, which is a good sanity check.
        for field in input_fields:
            assert isinstance(field, str), f"Input field '{field}' must be a string."
        for field in output_fields:
            assert isinstance(field, str), f"Output field '{field}' must be a string."
        # --- MODIFICATION END ---

        self.input_fields = input_fields
        self.output_fields = output_fields
        self.output_fields = output_fields

    def get_input_text(self, input_args: dict):
        for _ in self.input_fields:
            assert _ in input_args
        instruction = self.input_template.format_map(input_args)
        return [self.first_turn[0], instruction] if self.first_turn else [instruction]

    def get_output_text(self, output_args: dict):
        for _ in self.output_fields:
            assert _ in output_args
        response = self.output_template.format_map(output_args)
        return [self.first_turn[1], response] if self.first_turn else [response]


SeqRec_task_key = 'SeqRec'
SeqRanking_task_key = 'SeqRanking'

################################################################################################################
#                                                                                                              #
#                                sequential recommendation templates                                           #
#                                                                                                              #
################################################################################################################
SeqRec_CS_MR_group = {}

template = {
    'inst': "You need to generate a recommendation list considering user's preference from historical interactions." +
            "The historical interactions are provided as follows: {history}. " +
            "Please generate a recommendation list with {item_count} different items. " +
            "Each item should be enclosed by <SOI> and <EOI>. <SOI> should be generated before item title, and <EOI> should be generated after item title.",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(SeqRec_CS_MR_group)}",
}
SeqRec_CS_MR_group[f"{SeqRec_task_key}-{len(SeqRec_CS_MR_group)}"] = SFTTemplate(**template)

template = {
    'inst': "You need to select a recommendation list considering user's preference from historical interactions." +
            "The historical interactions are provided as follows: {history}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items. " +
            "Each item should be enclosed by <SOI> and <EOI>. <SOI> should be generated before item title, and <EOI> should be generated after item title.",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count', 'candidate_titles'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(SeqRec_CS_MR_group)}",
}
SeqRec_CS_MR_group[f"{SeqRec_task_key}-{len(SeqRec_CS_MR_group)}"] = SFTTemplate(**template)

template = {
    'inst': "Your task is generating a recommendation list according user's preference from historical interactions." +
            "The historical interactions are provided as follows: {history}. " +
            "Please generate a recommendation list with {item_count} different items.",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(SeqRec_CS_MR_group)}",
}
SeqRec_CS_MR_group[f"{SeqRec_task_key}-{len(SeqRec_CS_MR_group)}"] = SFTTemplate(**template)

template = {
    'inst': "Your task is selecting a recommendation list according user's preference from historical interactions." +
            "The historical interactions are provided as follows: {history}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items.",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count', 'candidate_titles'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(SeqRec_CS_MR_group)}",
}
SeqRec_CS_MR_group[f"{SeqRec_task_key}-{len(SeqRec_CS_MR_group)}"] = SFTTemplate(**template)


SeqRec_MR_group = {}

template = {
    'inst': "Your task is generating a recommendation list according user's preference from historical interactions." +
            "The historical interactions are provided as follows: {history}. " +
            "Please generate a recommendation list with {item_count} different items.",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(SeqRec_MR_group)}",
}
SeqRec_MR_group[f"{SeqRec_task_key}-{len(SeqRec_MR_group)}"] = SFTTemplate(**template)

template = {
    'inst': "Your task is selecting a recommendation list according user's preference from historical interactions." +
            "The historical interactions are provided as follows: {history}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items.",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count', 'candidate_titles'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(SeqRec_MR_group)}",
}
SeqRec_MR_group[f"{SeqRec_task_key}-{len(SeqRec_MR_group)}"] = SFTTemplate(**template)


################################################################################################################
#                                                                                                              #
#                                         val and test templates                                               #
#                                                                                                              #
################################################################################################################

ValSeqRec_group = {}
template = {
    'first_turn': ["You need to generate a recommendation list considering user's preference from historical interactions.",
                   "Ok, I will consider user's preference. "],
    'inst': "The historical interactions are provided as follows: {history}. " +
            "Could you recommend me a list {item_count} different items?",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(ValSeqRec_group)}",
}
ValSeqRec_group[f"{SeqRec_task_key}-{len(ValSeqRec_group)}"] = SFTTemplate(**template)


ValSeqRec_CS_group = {}
template = {
    'first_turn': ["You need to generate a recommendation list considering user's preference from historical interactions.",
                   "Ok, I will consider user's preference. "],
    'inst': "The historical interactions are provided as follows: {history}. " +
            "Could you recommend me a list {item_count} different items? " +
            "Each item should be enclosed by <SOI> and <EOI>. <SOI> should be generated before item title, and <EOI> should be generated after item title.",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(ValSeqRec_CS_group)}",
}
ValSeqRec_CS_group[f"{SeqRec_task_key}-{len(ValSeqRec_CS_group)}"] = SFTTemplate(**template)


ValSeqRec_CS_MR_group = {}
template = {
    'inst': "You need to generate a recommendation list considering user's preference from historical interactions. " +
            "The historical interactions are provided as follows: {history}. " +
            "Could you recommend me a list {item_count} different items? " +
            "Each item should be enclosed by <SOI> and <EOI>. <SOI> should be generated before item title, and <EOI> should be generated after item title.",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(ValSeqRec_CS_MR_group)}",
}
ValSeqRec_CS_MR_group[f"{SeqRec_task_key}-{len(ValSeqRec_CS_MR_group)}"] = SFTTemplate(**template)


ValSeqRec_MR_group = {}
template = {
    'inst': "You need to generate a recommendation list considering user's preference from historical interactions. " +
            "The historical interactions are provided as follows: {history}. " +
            "Could you recommend me a list {item_count} different items? ",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(ValSeqRec_MR_group)}",
}
ValSeqRec_MR_group[f"{SeqRec_task_key}-{len(ValSeqRec_MR_group)}"] = SFTTemplate(**template)

ValSeqRec_MR_same_group = {}

template = {
    'inst': "Your task is generating a recommendation list according user's preference from historical interactions." +
            "The historical interactions are provided as follows: {history}. " +
            "Please generate a recommendation list with {item_count} different items.",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRec_task_key}-{len(ValSeqRec_MR_same_group)}",
}
ValSeqRec_MR_same_group[f"{SeqRec_task_key}-{len(ValSeqRec_MR_same_group)}"] = SFTTemplate(**template)


ValSeqRanking_group = {}
template = {
    'first_turn': ["You need to generate a recommendation list considering user's preference from historical interactions. ",
                   "Ok, I will do it considering user's preference and candidate items. "],
    'inst': "The historical interactions are provided as follows: {history}. " +
            "The candidate items are: {candidate_titles}. " +
            "Please select a recommendation list with {item_count} different items from candidate items. ",
    'output': "{item_title_list}",
    'input_fields': ['history', 'item_count', 'candidate_titles'],
    'output_fields': ['item_title_list'],
    'template_id': f"{SeqRanking_task_key}-{len(ValSeqRanking_group)}",
}
ValSeqRanking_group[f"{SeqRanking_task_key}-{len(ValSeqRanking_group)}"] = SFTTemplate(**template)


if __name__ == '__main__':
    pass


################################################################################################################
#                                                                                                              #
#                                      Value Linking Templates                                                 #
#                                                                                                              #
################################################################################################################
ValueLinking_group = {}
template = {
    'inst': "Link the following mention to a correct database value: '{variation}'.",
    'output': "<SOI>{canonical_value}<EOI>",
    'input_fields': ['variation'],
    'output_fields': ['canonical_value'],
    'template_id': 'ValueLinking-1',
}
ValueLinking_group['ValueLinking-1'] = SFTTemplate(**template)

template = {
    'inst': "What is the canonical database value for the term '{variation}'?",
    'output': "<SOI>{canonical_value}<EOI>",
    'input_fields': ['variation'],
    'output_fields': ['canonical_value'],
    'template_id': 'ValueLinking-2',
}
ValueLinking_group['ValueLinking-2'] = SFTTemplate(**template)

# This template is for validation/testing where the output is just the target item.
ValValueLinking_group = {}
template = {
    'inst': "Link the following mention to a correct database value: '{variation}'.",
    'output': "<SOI>{canonical_value}<EOI>",
    'input_fields': ['variation'],
    'output_fields': ['canonical_value'],
    'template_id': 'ValValueLinking-1',
}
ValValueLinking_group['ValValueLinking-1'] = SFTTemplate(**template)