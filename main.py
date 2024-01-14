from getpass import getpass
from openai import OpenAI
from Document import Document


def build_api_request(prompt: str) -> list:
    api_request = [
        'llama-70b-chat',
        [
            {"role": "system", "content": "Assistant is a large language model that addresses every topic within the "
                                          "input text."},
            {"role": "user", "content": prompt},
        ]
    ]

    return api_request


def main():
    key = getpass('Insert api key: ')

    doc: Document = Document('docs/fitting_window.txt')

    print('Slicing the text...\n')

    slices: list[str] = doc.slice_document(2048)

    print(f'The document is composed of {doc.length} tokens.')
    print(f'It exceeds the context window.') if doc.length > 2048 else print(f'It fits the context window.')

    model = OpenAI(
        api_key=key,
        base_url='https://api.llama-api.com'
    )

    # feed the slices to the LLM and print the answers
    for s in slices:
        print('\nGenerating answer...')

        request: list = build_api_request(s)

        response = model.chat.completions.create(model=request[0], messages=request[1])

        print(f'--SLICE {slices.index(s) + 1}--\n {response.choices[0].message.content}')


if __name__ == '__main__':
    main()
