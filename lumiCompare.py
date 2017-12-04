import logging
import os
import argparse
import lumiValidate
from matplotlib import pyplot

lumiValidate.LUMI_TOP_YLIMIT = 800000

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    args = parse_predefined_args()

    fig_inst = make_compare_figure(args, 'Instantaneous', 'hz/ub')
    lumiValidate.reset_color_cycle()
    fig_intg = make_compare_figure(args, 'Integrated', '/ub')

    log.info('showing plots')
    pyplot.show()


def make_compare_figure(args, title, unit='hz/ub'):
    timerange = lumiValidate.parse_timerange_args(args)
    types = args.types
    if len(args.normtags) == 0 and len(args.types) == 0:
        types = ['pltzero', 'hfoc']
    lumis = types + args.normtags
    if len(lumis) < 2:
        raise IndexError('Too little (<2) lumi types and/or normtags provided')
    fig = pyplot.figure(figsize=lumiValidate.FIGURE_SIZE)
    fig.suptitle(title, fontsize=16)
    data = None
    data, cols = lumiValidate.get_avg_data(
        timerange, types, args.normtags, beam_selector=args.beams, unit=unit,
        merge_how='inner')
    if data is None:
        raise RuntimeError('Could not get data')
    x = lumis[0]
    y = lumis[1]
    if x in cols and y in cols:
        corr_plot = fig.add_subplot(3, 1, 3)
        lumiValidate.make_correlation_plot(corr_plot, data, x, y)
        rows = 3
        fig.subplots_adjust(**lumiValidate.FIGURE_ADJUSTS_TWO_FAT_ONE_SLIM_ROWS)
    else:
        log.error('no data for {} or/and {}: cannot make'
                  ' correlation plot'.format(x, y))
    lumi_plot = fig.add_subplot(rows, 1, 1)
    lumiValidate.make_lumi_plot(lumi_plot, data, cols, timerange, unit=unit)
    ratios_plot = fig.add_subplot(rows, 1, 2)
    lumiValidate.make_ratio_plot(ratios_plot, data, cols, timerange)

    return fig


def parse_predefined_args():
    description = '''Compare two lumis'''
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
        ' specified): pltzero hfoc')
    parser.add_argument(
        '--normtags', dest='normtags', metavar='normtag', type=str, nargs='+',
        default=[], help='space delimited normtags (file path also possible)')
    parser.add_argument(
        '-b', dest='beams', type=str, help='beam mode')
    return parser.parse_args()


if __name__ == '__main__':
    main()
