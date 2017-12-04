#!/usr/bin/env python
import logging
import os
import subprocess
import argparse
import itertools
import tempfile
import pandas
import numpy
from matplotlib import transforms, pyplot, ticker


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

RATIOS_TOP_YLIMIT = 1.25
RATIOS_BOTTOM_YLIMIT = 0.75
LUMI_TOP_YLIMIT = 11000
LUMI_BOTTOM_YLIMIT = 0
# default figure size (tuple) in inches
FIGURE_SIZE = (14, 11)
LUMI_UNIT = 'hz/ub'

# (Jonas): tight_layout would be way better, but it does not handle
# legends outside plots, so we have to do the following manual adjustments:
# adjust for figuresize depending on layout (also keeping in mind
# rotated xaxis ticks)
FIGURE_ADJUSTS_TWO_FAT_ROWS = {
    'top': 0.94, 'left': 0.07, 'right': 0.85, 'bottom': 0.15, 'hspace': 0.55}
FIGURE_ADJUSTS_TWO_FAT_ONE_SLIM_ROWS = {
    'top': 0.94, 'left': 0.07, 'right': 0.85, 'bottom': 0.04, 'hspace': 0.80}
XAXIS_TICKS_MAX = 40
SPECIAL_COLOR1 = '#FF0000'
BG_STRIPE_ALPHA = 0.1
BG_STRIPE_COLOR = '#0000FF'
COLOR_LIST = [
    '#000000', '#00FF00', '#0000FF', '#01FFFE', '#FFA6FE', '#FFDB66',
    '#006401', '#010067', '#95003A', '#007DB5', '#FF00F6', '#774D00',
    '#90FB92', '#0076FF', '#D5FF00', '#FF937E', '#6A826C', '#FF029D',
    '#FE8900', '#7A4782', '#7E2DD2', '#85A900', '#FF0056', '#A42400',
    '#00AE7E', '#683D3B', '#BDC6FF', '#263400', '#BDD393', '#00B917',
    '#9E008E', '#001544', '#C28C9F', '#FF74A3', '#01D0FF', '#004754',
    '#E56FFE', '#788231', '#0E4CA1', '#91D0CB', '#BE9970', '#968AE8',
    '#BB8800', '#43002C', '#DEFF74', '#00FFC6', '#FFE502', '#620E00',
    '#008F9C', '#98FF52', '#7544B1', '#B500FF', '#00FF78', '#FF6E41',
    '#005F39', '#6B6882', '#5FAD4E', '#A75740', '#A5FFD2', '#FFB167',
    '#009BFF', '#E85EBE'
]
COLORS = itertools.cycle(COLOR_LIST)


def main():
    parser = predefined_arg_parser()
    log.info('parsing main arguments')
    args = parser.parse_args()
    timerange = parse_timerange_args(args)
    types = args.types
    if len(args.normtags) == 0 and len(args.types) == 0:
        types = ['hfoc', 'bcm1f', 'pltzero', 'online']

    fig = pyplot.figure(figsize=FIGURE_SIZE)
    data = None
    if args.xing:
        plot = fig.add_subplot(111)
        ltype = args.types[0] if args.types else None
        normtag = args.normtags[0] if args.normtags else None
        data, bxd_cols, name = get_bunch_data(
            timerange, ltype, normtag, args.beams, unit=LUMI_UNIT)
        if data is None:
            raise RuntimeError('Could not get data')
        make_bunch_plot(plot, data, bxd_cols, timerange, name, args.threshold,
                        unit=LUMI_UNIT)
        fig.tight_layout()
    else:
        if args.single_bunch is not None:
            data, cols = get_single_bunch_data(
                timerange, args.single_bunch, types, args.normtags, args.beams,
                unit=LUMI_UNIT)
        else:
            data, cols = get_avg_data(
                timerange, types, args.normtags, args.beams, unit=LUMI_UNIT)
        if data is None:
            raise RuntimeError('Could not get data')
        fig.subplots_adjust(**FIGURE_ADJUSTS_TWO_FAT_ROWS)
        rows = 2
        if args.correlate is not None:
            x = args.correlate[0]
            y = args.correlate[1]
            if x in cols and y in cols:
                corr_plot = fig.add_subplot(3, 1, 3)
                make_correlation_plot(corr_plot, data, x, y)
                rows = 3
                fig.subplots_adjust(**FIGURE_ADJUSTS_TWO_FAT_ONE_SLIM_ROWS)
            else:
                log.error('no data for {} or/and {}: cannot make'
                          ' correlation plot'.format(x, y))
        lumi_plot = fig.add_subplot(rows, 1, 1)
        make_lumi_plot(lumi_plot, data, cols, timerange, unit=LUMI_UNIT,
                       single_bunch=args.single_bunch)
        ratios_plot = fig.add_subplot(rows, 1, 2)
        make_ratio_plot(ratios_plot, data, cols, timerange, args.primary)

    if args.outfile is not None:
        log.info('printing data to file %s', args.outfile)
        data.to_csv(args.outfile)

    log.info('showing plots')
    pyplot.show()


def predefined_arg_parser():
    description = '''
    Tool for plotting luminosity values from brilcalc. brilcalc must be
    setup before using this script. Also do not forget X Window forwarding
    if you run this script remotely, e.g.: ssh <user>@lxplus -X'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-r', dest='run', type=int, help='run number')
    parser.add_argument(
        '-f', dest='fill', type=int, help='fill number')
    parser.add_argument(
        '--begin', dest='begin', type=str, help='begin')
    parser.add_argument(
        '--end', dest='end', type=str, help='end')
    parser.add_argument(
        '-i', dest='input_select', type=str, help='input selection json')
    parser.add_argument(
        '--types', dest='types', metavar='type', type=str, nargs='+',
        default=[], help='space delimited types. Default (if no normtags'
        ' specified): hfoc bcm1f pltzero online')
    parser.add_argument(
        '--normtags', dest='normtags', metavar='normtag', type=str, nargs='+',
        default=[], help='space delimited normtags (file path also possible)')
    parser.add_argument(
        '--correlate', dest='correlate', type=str, nargs=2,
        help='x and y: two types (or normtags) for correlation plot')
    parser.add_argument(
        '-b', dest='beams', type=str, help='beam mode')
    parser.add_argument(
        '--single-bunch', dest='single_bunch', type=int,
        help='make plots for specific bunch')
    parser.add_argument(
        '--xing', dest='xing', action='store_true',
        help='bx granurality (first type or normtag is selected if any)')
    parser.add_argument(
        '-o', dest='outfile', type=str, help='file to output data')
    parser.add_argument(
        '-t', dest='threshold', type=float, default=0.4,
        help='values < max*threshold are not included in plots (this option'
        ' applies for bunch plots (--xing) only). Default: 0.4')
    parser.add_argument(
        '--primary', dest='primary', metavar='type/normtag', type=str,
        nargs='+', default=[],
        help='list of space-delimited primary luminometer. If specified, '
        'only ratios to at least one primary luminometer will be shown')
    return parser


def parse_timerange_args(args):
    timerange = {}
    if args.run is not None:
        timerange['run'] = args.run
    elif args.fill is not None:
        timerange['fill'] = args.fill
    elif args.input_select is not None:
        timerange['input_select'] = args.input_select
    elif args.begin is not None and args.end is not None:
        timerange['begin'] = args.begin
        timerange['end'] = args.end
    else:
        raise ValueError('Either run, fill or begin+end must by specified')
    return timerange

# ---------
# retrieving and preparing data
# ---vvv---


def get_avg_data(timerange, types=[], normtags=[], beam_selector=None,
                 unit='hz/ub', merge_how='outer'):

    log.info('get avg data')
    found_columns = []
    merged = None
    for request in types + normtags:
        options = []
        if request in types:
            if (request != 'online'):
                options += ['--type', request]
        elif request in normtags:
            options += ['--normtag', request]
            if os.path.isfile(request):
                log.info('normtag is file: %s', request)
                request = 'normtag'

        data = call_brilcalc(timerange, options, beam_selector=beam_selector,
                             unit=unit)
        if data is None:
            log.error('Failed fetching data for {}. Skipping'.format(request))
            continue
        found_columns.append(request)

        value_field_name = 'delivered(' + unit + ')'
        data[value_field_name] = data[value_field_name].map(float)
        data.rename(columns={value_field_name: request}, inplace=True)
        data = data.loc[:, ('#run:fill', 'ls', request)]

        if merged is None:
            merged = data
        else:
            merged = pandas.merge(merged, data, how=merge_how,
                                  on=['#run:fill', 'ls'])

    merged = format_dataframe(merged)
    log.info('sucessfully got data')
    return merged, found_columns


def get_single_bunch_data(
        timerange, bxid, types=[], normtags=[], beam_selector=None,
        unit='hz/ub', merge_how='outer'):
    log.info('get single bunch data')
    found_columns = []
    merged = None
    for request in types + normtags:

        def bx_to_col_fn(row):
            bunch_data = row['bunches'][1:-1].split(' ')
            try:
                index = bunch_data[0::3].index(str(bxid))
            except ValueError:
                log.warning('No bunch {} for run:{} LS:{}'.format(
                    bxid, row['run'], row['ls']))
                delivered = None
            else:
                delivered = float(bunch_data[1::3][index])
            return pandas.Series(data=[delivered], index=[request])

        if request in types:
            data, cols, name = get_bunch_data(
                timerange, request, None, beam_selector, bx_to_col_fn,
                unit=unit)
        elif request in normtags:
            data, cols, name = get_bunch_data(
                timerange, None, request, beam_selector, bx_to_col_fn,
                unit=unit)
        if data is None:
            log.error('Failed fetching data for {}. Skipping'.format(request))
            continue
        found_columns.append(request)

        if merged is None:
            merged = data
        else:
            merged = pandas.merge(merged, data, how=merge_how,
                                  on=['fill', 'run', 'ls'])

    log.info('sucessfully got data')
    return merged, found_columns


def format_dataframe(data):
    # cleanup rows where #run:fill is corrupted
    data = data[data['#run:fill'].str.contains(':')].copy()
    data['ls'] = data['ls'].map(int_before_colon).astype(int)
    data['run'] = data['#run:fill'].map(int_before_colon).astype(int)
    data['fill'] = data['#run:fill'].map(int_after_colon).astype(int)
    data.drop('#run:fill', axis=1, inplace=True)
    data.sort_values(['run', 'ls'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def call_brilcalc(timerange, options, beam_selector=None, unit='hz/ub'):
    cmd = make_brilcalc_call_template(timerange, beam_selector, unit)
    cmd += options
    f = tempfile.NamedTemporaryFile()
    log.debug('temp file name: %s', f.name)
    cmd += ['-o', f.name]

    log.info('calling subprocess: {}'.format(' '.join(cmd)))
    ret_code = subprocess.call(cmd)
    if ret_code != 0:
        log.error('subprocess terminated with'
                  'non-zero return code: {}'.format(ret_code))
        return None

    # data = data[:-3] # not good when file contains '#Check JSON',
    # therefor we do manual clenaup
    clean_data_file_footer(f)

    data = pandas.read_csv(f.name, skiprows=1)
    if data.empty:
        log.error('No data parsed for {}'.format(' '.join(cmd)))
        return None
    f.close()
    return data


def clean_data_file_footer(f):
    f.seek(0)
    cutoffpos = 0
    line = f.readline()
    while line != '':
        if line.startswith('#Summary:'):
            break
        cutoffpos = f.tell()
        line = f.readline()
    f.truncate(cutoffpos)


def all_bx_to_cols(row):
    bunch_data = row['bunches'][1:-1].split(' ')
    cols = ['bxd:' + str(id) for id in bunch_data[0::3]]
    delivereds = [float(delivered) for delivered in bunch_data[1::3]]
    return pandas.Series(data=delivereds, index=cols)


def get_bunch_data(timerange, ltype=None, normtag=None, beam_selector=None,
                   bx_to_cols_fn=all_bx_to_cols, unit='hz/ub'):
    log.info('get all bunch data')
    name = 'online'
    options = ['--xing']
    if ltype is not None:
        if ltype != 'online':
            options += ['--type', str(ltype)]
            name = str(ltype)
    elif normtag is not None:
        options += ['--normtag', str(normtag)]
        name = str(normtag)
        if os.path.isfile(name):
            log.info('normtag is file: {}'.format(name))
            name = os.path.basename(name)
    data = call_brilcalc(timerange, options, beam_selector=None)
    if data is None:
        log.error('Failed fetching data for {} bunches'.format(name))
        return None, None, None

    bx_col_name = '[bxidx bxdelivered(' + unit + ') bxrecorded(' + unit + ')]'
    data.rename(inplace=True, columns={bx_col_name: 'bunches'})
    data = data.loc[:, ('#run:fill', 'ls', 'bunches')]
    data = format_dataframe(data)

    split_bunches = data.apply(bx_to_cols_fn, axis=1)
    data.drop('bunches', axis=1, inplace=True)
    data = pandas.concat([data, split_bunches], axis=1)
    bxd_cols = data.columns.values.tolist()[3:]
    log.info('sucessfully got data')
    return data, bxd_cols, name


def make_brilcalc_call_template(timerange, beam_selector=None, unit='hz/ub'):
    if timerange is None:
        raise ValueError('timerange cannot be None')
    time_selection = None
    if 'run' in timerange:
        time_selection = ['-r', str(timerange['run'])]
    elif 'fill' in timerange:
        time_selection = ['-f', str(timerange['fill'])]
    elif 'input_select' in timerange:
        time_selection = ['-i', str(timerange['input_select'])]
    elif 'begin' in timerange and 'end' in timerange:
        time_selection = ['--begin', str(timerange['begin']),
                          '--end', str(timerange['end'])]
    if time_selection is None:
        raise ValueError(
            'Either run, fill, begin+end, or input json must by specified')

    cmd = ['brilcalc', 'lumi', '--byls', '-u', unit]
    cmd += time_selection
    if beam_selector is not None:
        cmd += ['-b', beam_selector]
    return cmd


def int_before_colon(value):
    return int(value.split(':')[0])


def int_after_colon(value):
    return int(value.split(':')[1])

# ---------
# plotting
# ---vvv---


def make_lumi_plot(plot, data, cols, timerange, unit, single_bunch=None):
    log.info('making lumi plot')
    plot_by_columns(plot, data, cols, 'online')
    plot.set_ylabel('lumi (' + unit + ')')
    ylims = plot.get_ylim()
    if ylims[0] < LUMI_BOTTOM_YLIMIT:
        plot.set_ylim(bottom=LUMI_BOTTOM_YLIMIT)
    if ylims[1] > LUMI_TOP_YLIMIT:
        plot.set_ylim(top=LUMI_TOP_YLIMIT)

    if single_bunch is None:
        pre_name = 'Average luminosity'
    else:
        pre_name = 'Bunch #{} luminosity'.format(single_bunch)
    plot.set_title(pre_name + '. ' + get_title_timerange_part(timerange, data))

    auto_separate_chunks_on_plot(plot, data, timerange)


def make_ratio_plot(plot, data, cols, timerange, primary_luminometers=[]):
    ratios = calculate_ratios(data, cols, primary_luminometers)
    log.info('creating ratios plot')
    plot_by_columns(plot, data, ratios)
    plot.set_title('Lumi ratios')
    plot.set_ylabel('lumi ratios')
    ylims = plot.get_ylim()
    if ylims[0] < RATIOS_BOTTOM_YLIMIT:
        plot.set_ylim(bottom=RATIOS_BOTTOM_YLIMIT)
    if ylims[1] > RATIOS_TOP_YLIMIT:
        plot.set_ylim(top=RATIOS_TOP_YLIMIT)

    auto_separate_chunks_on_plot(plot, data, timerange)


def make_correlation_plot(plot, data, x, y):
    log.info('making correlation plot')
    plot.scatter(data[x], data[y], alpha=0.5)
    log.info('calculating fit')
    # filter NaN's None's ...
    mask = numpy.isfinite(data[x]) & numpy.isfinite(data[y])
    k, c = numpy.polyfit(data[x][mask], data[y][mask], 1)
    linex = [data[x].min(), data[x].max()]
    liney = [data[x].min()*k+c, data[x].max()*k+c]
    log.info('adding fit line')
    plot.plot(linex, liney, c=SPECIAL_COLOR1)
    text = r'$y(x)= x*{0} + ({1})$'.format(k, c)
    plot.text(0.0, 1.0, s=text, ha='left', va='top',
              fontsize=16, transform=plot.transAxes)
    plot.set_xlabel(x)
    plot.set_ylabel(y)
    plot.set_title('Correlation')
    plot.grid(True)

def make_bunch_plot(plot, data, cols, timerange, name, threshold, unit='hz/ub'):

    def filter_by_percentage_of_max(row):
        rmax = row.max()
        row[row < rmax*threshold] = None
        return row

    data.iloc[:, 3:] = data.iloc[:, 3:].apply(
        filter_by_percentage_of_max, axis=1)

    log.info('creating plot')
    log.info('plotting by bunch')
    plot_by_columns(plot, data, cols, legend=False)
    plot.set_ylabel('bxdelivered (' + unit + ')')

    plot.set_title(name + ' per bunch. ' +
                   get_title_timerange_part(timerange, data) +
                   ', threshold: ' + str(threshold))
    auto_separate_chunks_on_plot(plot, data, timerange)


def plot_by_columns(subplot, data, cols, special=None, legend=True):
    log.info('plotting by column')
    for col in cols:
        linestyle = '-'
        color = COLORS.next()
        if (col == special):
            color = SPECIAL_COLOR1
            linestyle = '--'

        log.debug('adding line for: %s', col)
        subplot.plot(data.index, data[col].values, linestyle=linestyle,
                     c=color, label=col)

    if legend:
        subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    subplot.set_xlabel('RUN:LS')
    subplot.xaxis.set_major_formatter(create_runls_ticks_formatter(data))
    subplot.xaxis.set_major_locator(ticker.MaxNLocator(nbins=XAXIS_TICKS_MAX))
    subplot.xaxis.set_tick_params(labelsize=12)
    subplot.grid(True)
    # (Jonas): could not find more beautiful way to set axis labels rotation
    pyplot.setp(subplot.xaxis.get_majorticklabels(), rotation=90)


def calculate_ratios(data, cols, primary_luminometers=[]):
    '''Update DataFrame 'data' with ratios'''
    log.info('calculating ratios')
    # back up numpy settings and set to ignore all errors (to handle
    # 'None's and division by zero)
    old_numpy_settings = numpy.seterr(all='ignore')
    comparables = [x for x in cols if x != 'online']
    if len(primary_luminometers):
        comparables = [x for x in comparables if x in primary_luminometers]
    calculated_ratios = []
    for above_idx, above in enumerate(comparables):
        for below in comparables[above_idx + 1:]:
            name = above + '/' + below
            data[name] = data[above]/data[below]
            calculated_ratios.append(name)

    numpy.seterr(**old_numpy_settings)
    return calculated_ratios


# ---------
# decorations and helpers
# ---vvv---


def get_fill_by_run(runnr, data):
    return int(data[data['run'] == runnr].iloc[0]['fill'])


def get_title_timerange_part(timerange, data):
    if 'run' in timerange:
        return 'Run {}, Fill {}'.format(
            timerange['run'], get_fill_by_run(timerange['run'], data))
    elif 'fill' in timerange:
        return 'Fill {}'.format(timerange['fill'])
    elif 'input_select' in timerange:
        return 'Custom input JSON'
    else:
        return 'Begin {}, End {}'.format(timerange['begin'], timerange['end'])


def auto_separate_chunks_on_plot(plot, data, timerange):
    if 'run' in timerange:
        return # do not separate
    elif 'fill' in timerange:
        separate_chunks_on_plot(plot, data, chunk='run')
    else: # begin+end or input_select case
        separate_chunks_on_plot(plot, data, chunk='fill')


def create_runls_ticks_formatter(dataframe):
    log.info('creating x axis ticks formatter')
    labels = ['{0:d}:{1:>4}'.format(run, ls)
              for run, ls
              in zip(dataframe['run'], dataframe['ls'])]

    def runnr_lsnr_ticks(x, p):
        x = int(x)
        if x >= len(labels) or x < 0:
            return ''
        else:
            return labels[x]

    log.info('ticks formatter created')
    return ticker.FuncFormatter(runnr_lsnr_ticks)


def separate_chunks_on_plot(subplot, data, chunk='run'):
    '''
    put number every chunk change and color background for every second chunk

    chunk: 'run' or 'fill'
    '''
    log.info('visualizing run changes')
    # prepare transformation to treat x as data values and y as
    # relative plot position
    trans = transforms.blended_transform_factory(
        subplot.transData, subplot.transAxes)
    put_bg_switch = False
    for chunk_group in data.groupby(chunk):
        chunk_number, data = chunk_group
        x = data.index[0]
        subplot.text(x, 0.99, s=chunk_number, ha='left', va='top',
                     rotation=90, transform=trans)
        if put_bg_switch:
            xmax = data.index[-1]
            subplot.axvspan(x, xmax, facecolor=BG_STRIPE_COLOR,
                            alpha=BG_STRIPE_ALPHA)

        put_bg_switch = not put_bg_switch


def reset_color_cycle():
    global COLORS
    COLORS = itertools.cycle(COLOR_LIST)


if __name__ == '__main__':
    main()
