CH_input = 1
CH_output = 90
CH_basic = 16*4
CH_inter = 64*4
freq_bins = 201
win_frames = 11
act_inplace = False
negative_slope = 0.02
pars = {
    'encoder': {
        'layer_1': {
            'conv': {
                'in_channels': CH_input,
                'out_channels': CH_basic,
                'kernel_size': (3, 1),
                'stride': (1, 1),
                'padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_2': {
            'conv': {
                'in_channels': CH_basic,
                'out_channels': CH_basic,
                'kernel_size': (1, 3),
                'stride': (1, 1),
                'padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_3': {
            'conv': {
                'in_channels': CH_basic,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_4': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_5': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_6': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_7': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 1),
                'stride': (2, 1),
                'padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_8': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (1, 3),
                'stride': (2, 1),
                'padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_9': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (1, 1),  # (2, 2)
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_10': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_inter,
                'kernel_size': (1, 1),
                'stride': (1, 1),
                'padding': (0, 0)
            },
            'bn': {
                'num_features': CH_inter
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        }
    },
    'decoder': {
        'layer_1': {
            'tconv': {
                'in_channels': CH_inter,
                'out_channels': CH_basic * 2,
                'kernel_size': (1, 1),
                'stride': (1, 1),
                'padding': (0, 0),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_2': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (1, 1),   # (2, 2)
                'padding': (1, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_3': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (1, 3),
                'stride': (2, 1),
                'padding': (0, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_4': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 1),
                'stride': (2, 1),
                'padding': (1, 0),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_5': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_6': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_7': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_8': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_9': {
            'tconv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic,
                'kernel_size': (1, 3),
                'stride': (1, 1),
                'padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_10': {
            'tconv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_input,
                'kernel_size': (3, 1),
                'stride': (1, 1),
                'padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 2  # not used
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
    }
}


pars_new = {
    'encoder': {
        'layer_1': {
            'conv': {
                'in_channels': CH_input,
                'out_channels': CH_basic,
                'kernel_size': (7, 3),
                'stride': (1, 1),
                'padding': (3, 1)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_2': {
            'conv': {
                'in_channels': CH_basic,
                'out_channels': CH_basic,
                'kernel_size': (5, 3),
                'stride': (1, 1),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_3': {
            'conv': {
                'in_channels': CH_basic,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_4': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_5': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_6': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_7': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_8': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_9': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (1, 1),  # (2, 2)
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_10': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_inter,
                'kernel_size': (1, 1),
                'stride': (1, 1),
                'padding': (0, 0)
            },
            'bn': {
                'num_features': CH_inter
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        }
    },
    'decoder': {
        'layer_1': {
            'tconv': {
                'in_channels': CH_inter,
                'out_channels': CH_basic * 2,
                'kernel_size': (1, 1),
                'stride': (1, 1),
                'padding': (0, 0),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_2': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (1, 1),   # (2, 2)
                'padding': (1, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_3': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_4': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_5': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_6': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_7': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_8': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_9': {
            'tconv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic,
                'kernel_size': (5, 3),
                'stride': (1, 1),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
        'layer_10': {
            'tconv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_input,
                'kernel_size': (7, 3),
                'stride': (1, 1),
                'padding': (3, 1)
            },
            'bn': {
                'num_features': CH_basic * 2  # not used
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': act_inplace
            }
        },
    }
}

pars_mode1 = {
    'mode': 1,
    'encoder': {
        'layer_1': {
            'conv': {
                'in_channels': CH_input,
                'out_channels': CH_basic,
                'kernel_size': (7, 3),
                'stride': (1, 1),
                'padding': (3, 1)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_2': {
            'conv': {
                'in_channels': CH_basic,
                'out_channels': CH_basic,
                'kernel_size': (7, 3),
                'stride': (1, 1),
                'padding': (3, 1)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_3': {
            'conv': {
                'in_channels': CH_basic,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_4': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_5': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_6': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_7': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 1),
                'stride': (2, 1),
                'padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_8': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (1, 1),
                'stride': (2, 1),
                'padding': (0, 0)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_9': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (1, 1),
                'stride': (1, 1),  # (2, 2)
                'padding': (0, 0)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_10': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_inter,
                'kernel_size': (1, 1),
                'stride': (1, 1),
                'padding': (0, 0)
            },
            'bn': {
                'num_features': CH_inter
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        }
    },
    'decoder': {
        'layer_1': {
            'tconv': {
                'in_channels': CH_inter,
                'out_channels': CH_basic * 2,
                'kernel_size': (1, 1),
                'stride': (1, 1),
                'padding': (0, 0),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_2': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (1, 1),
                'stride': (1, 1),   # (2, 2)
                'padding': (0, 0),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_3': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (1, 1),
                'stride': (2, 1),
                'padding': (0, 0),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_4': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 1),
                'stride': (2, 1),
                'padding': (1, 0),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_5': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_6': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_7': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_8': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_9': {
            'tconv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic,
                'kernel_size': (7, 3),
                'stride': (1, 1),
                'padding': (3, 1)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_10': {
            'tconv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_output,
                'kernel_size': (7, 3),
                'stride': (1, 1),
                'padding': (3, 1)
            },
            'bn': {
                'num_features': CH_output  # not used
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
    },
    'att': {
        'in_features': freq_bins,
        'out_features': 1
    },
    'trans_layer': {
        'in_channels': CH_output,
        'out_channels': CH_input,
        'kernel_size': (1, win_frames),
    }
}


CH_complexity = 45

pars_gh = {
    'mode': 1,
    'encoder': {
        'layer_1': {
            'conv': {
                'in_channels': CH_input,
                'out_channels': CH_complexity,
                'kernel_size': (7, 1),
                'stride': (1, 1),
                'padding': (3, 0)
            },
            'bn': {
                'num_features': CH_complexity
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_2': {
            'conv': {
                'in_channels': CH_complexity,
                'out_channels': CH_complexity,
                'kernel_size': (1, 7),
                'stride': (1, 1),
                'padding': (0, 3)
            },
            'bn': {
                'num_features': CH_complexity
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_3': {
            'conv': {
                'in_channels': CH_complexity,
                'out_channels': CH_complexity * 2,
                'kernel_size': (6, 4),
                'stride': (2, 2),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_4': {
            'conv': {
                'in_channels': CH_complexity * 2,
                'out_channels': CH_complexity * 2,
                'kernel_size': (7, 5),
                'stride': (2, 1),
                'padding': (3, 2)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_5': {
            'conv': {
                'in_channels': CH_complexity * 2,
                'out_channels': CH_complexity * 2,
                'kernel_size': (5, 3),
                'stride': (2, 2),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_6': {
            'conv': {
                'in_channels': CH_complexity * 2,
                'out_channels': CH_complexity * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_7': {
            'conv': {
                'in_channels': CH_complexity * 2,
                'out_channels': CH_complexity * 2,
                'kernel_size': (5, 3),
                'stride': (2, 2),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_8': {
            'conv': {
                'in_channels': CH_complexity * 2,
                'out_channels': CH_complexity * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_9': {
            'conv': {
                'in_channels': CH_complexity * 2,
                'out_channels': CH_complexity * 2,
                'kernel_size': (5, 3),
                'stride': (2, 2),  # (2, 2)
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_10': {
            'conv': {
                'in_channels': CH_complexity * 2,
                'out_channels': 128,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': 128
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        }
    },
    'decoder': {
        'layer_1': {
            'tconv': {
                'in_channels': 128,
                'out_channels': CH_complexity * 2,
                'kernel_size': (4, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_2': {
            'tconv': {
                'in_channels': CH_complexity * 4,
                'out_channels': CH_complexity * 2,
                'kernel_size': (4, 2),
                'stride': (2, 2),   # (2, 2)
                'padding': (1, 0),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_3': {
            'tconv': {
                'in_channels': CH_complexity * 4,
                'out_channels': CH_complexity * 2,
                'kernel_size': (4, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_4': {
            'tconv': {
                'in_channels': CH_complexity * 4,
                'out_channels': CH_complexity * 2,
                'kernel_size': (4, 2),
                'stride': (2, 2),
                'padding': (1, 0),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_5': {
            'tconv': {
                'in_channels': CH_complexity * 4,
                'out_channels': CH_complexity * 2,
                'kernel_size': (4, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_6': {
            'tconv': {
                'in_channels': CH_complexity * 4,
                'out_channels': CH_complexity * 2,
                'kernel_size': (4, 2),
                'stride': (2, 2),
                'padding': (1, 0),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_7': {
            'tconv': {
                'in_channels': CH_complexity * 4,
                'out_channels': CH_complexity * 2,
                'kernel_size': (6, 3),
                'stride': (2, 1),
                'padding': (2, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_8': {
            'tconv': {
                'in_channels': CH_complexity * 4,
                'out_channels': CH_complexity * 2,
                'kernel_size': (7, 5),
                'stride': (2, 2),
                'padding': (2, 1),
                # 'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_9': {
            'tconv': {
                'in_channels': CH_complexity * 3,
                'out_channels': CH_complexity * 2,
                'kernel_size': (1, 7),
                'stride': (1, 1),
                'padding': (0, 3)
            },
            'bn': {
                'num_features': CH_complexity * 2
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
        'layer_10': {
            'tconv': {
                'in_channels': CH_complexity * 3,
                'out_channels': CH_output,
                'kernel_size': (7, 1),
                'stride': (1, 1),
                'padding': (3, 0)
            },
            'bn': {
                'num_features': CH_output  # not used
            },
            'act': {
                'negative_slope': negative_slope,
                'inplace': act_inplace
            }
        },
    },
    'att': {
        'in_features': freq_bins,
        'out_features': 1
    },
    'trans_layer': {
        'in_channels': CH_output,
        'out_channels': CH_input,
        'kernel_size': (1, 1),
    }
}

