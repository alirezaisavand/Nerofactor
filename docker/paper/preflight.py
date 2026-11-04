"""Future host-side preflight; validates a context before issuing its checksum gate.

Requires an available host Python >=3.8, never Python in the historical CUDA base.
Does not invoke installers, Docker, source exports or models.
"""

import argparse
from pathlib import Path

from validate import DEFINITION_FILES, HERE, definition, digest, inputs, inventory, require


RECEIPT = 'PREFLIGHT.sha256'
BASE_FILE = 'PREFLIGHT.base-image'


def prepare(context, base):
    before = inventory(context)
    reference = definition()
    expected_definition = {name: digest(HERE / name) for name in DEFINITION_FILES}
    require(inventory(context / 'definition') == expected_definition,
            'Staged definition inventory/bytes differ from the reviewed definition')
    require(digest(context / 'Dockerfile') == expected_definition['Dockerfile'],
            'Context Dockerfile differs from reviewed definition')
    require(digest(context / '.dockerignore') == digest(HERE / '.dockerignore'),
            'Context ignore rules differ from reviewed definition')
    # The complete existing validation boundary, not just the Debian checksums.
    inputs(context / 'inputs', context / 'workspace', base, reference)
    expected = {'Dockerfile': expected_definition['Dockerfile'],
                '.dockerignore': digest(HERE / '.dockerignore')}
    for folder in ['definition', 'inputs', 'workspace']:
        expected.update({folder + '/' + name: sha
                         for name, sha in inventory(context / folder).items()})
    require(inventory(context) == expected == before,
            'Context has extra files or an existing preflight receipt; use a fresh context')
    # Receipt anchors resolved.json itself, definitions, ignore rules and every artifact.
    # Its digest must be reviewed/supplied outside the context as the required build ARG.
    with (context / BASE_FILE).open('x') as stream:
        stream.write(base + '\n')
    expected[BASE_FILE] = digest(context / BASE_FILE)
    receipt = context / RECEIPT
    with receipt.open('x') as stream:
        stream.write(''.join('{}  ./{}\n'.format(sha, name)
                             for name, sha in sorted(expected.items())))
    return digest(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--base-image', required=True)
    args = parser.parse_args()
    try:
        sha = prepare(args.context, args.base_image)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, 'Preflight rejected: {}\n'.format(exc))
    print('PAPER_PREFLIGHT_SHA256=' + sha)
    print('Review/save this digest outside the context; supply it explicitly in the later approved build.')


if __name__ == '__main__':
    main()
