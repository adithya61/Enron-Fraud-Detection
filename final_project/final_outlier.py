"""
    remove one max value that seems like a outlier
"""


def remove_outlier(dictionary, features):
    """
    remove an outlier for one call
    :param dictionary:
    :param features: List of all the features used
    :return: dictionary with outlier removed
    """
    final_dict = dictionary
    # print(final_dict['TOTAL'])
    for item in features:
        maxim = -1
        maxim = float(maxim)
        name = ''
        # print('feature', item)
        # print('-----------------------')
        for key in dictionary:
            cmp = dictionary[key][item]
            if cmp > maxim and cmp != 'NaN':
                maxim = cmp
                name = key
                # print('max value', maxim)

    del (final_dict[name])
    # print('removed name', name)

    # print('all', features)
    return final_dict

