import click
import pandas as pd


@click.group()
def cli():
    pass

@cli.command()
@click.option('--input_file', type=click.Path(exists=True))
@click.option('--output_file', default='inputs.txt')
def preprocess(input_file, output_file, testing_mode=True):
    """
    python generate_inputs.py preprocess --input_file <input_file> --output_file <output_file>
    """

    # Read the input file
    data = pd.read_excel(input_file)
    data.rename(
        columns={
            'sexPulse': 'sex',
            'agePulse': 'age',
            'appointmentTime': 'time'
        },
        inplace=True
    )
    data['time'] = data['time'].astype(str)
    data['time'] = pd.to_timedelta(data['time']).dt.total_seconds() / (60 * 60)  # Convert to hours

    # Keep preprocessed data table
    data.to_csv(output_file, sep='\t', index=False)


@cli.command()
@click.option('--input_file', type=click.Path(exists=True))
@click.option('--output_file', type=click.Path(), default='responses.txt')
def responses(input_file, output_file, testing_mode=False):
    """
    python generate_inputs.py responses --input_file <input_file> --output_file <output_file>
    """

    # Read the input file
    data = pd.read_csv(input_file, sep='\t')
    df = {'response': []}
    count = 0
    for col in data.columns:
        if col not in ['identifier', 'sex', 'age', 'time']:
            if testing_mode and (count > 2):
                break
            df['response'].append(col)
            count += 1
    pd.DataFrame(df).to_csv(output_file, header=False, index=False)


if __name__ == '__main__':

    cli()
