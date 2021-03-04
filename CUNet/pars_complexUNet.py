CH_input = 1
CH_basic = 16
CH_inter = 64

pars = {
    'encoder': {
        'layer_1': {
            'conv': {
                'in_channels': CH_input,
                'out_channels': CH_basic,
                'kernel_size': (7, 1),
                'stride': (1, 1),
                'padding': (3, 0)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_2': {
            'conv': {
                'in_channels': CH_basic,
                'out_channels': CH_basic,
                'kernel_size': (1, 7),
                'stride': (1, 1),
                'padding': (0, 3)
            },
            'bn': {
                'num_features': CH_basic
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_3': {
            'conv': {
                'in_channels': CH_basic,
                'out_channels': CH_basic * 2,
                'kernel_size': (7, 5),
                'stride': (2, 2),
                'padding': (3, 2)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_4': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (7, 5),
                'stride': (2, 1),
                'padding': (3, 2)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_5': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 2),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_6': {
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
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_7': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 2),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_8': {
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
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_9': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),  # (2, 2)
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_10': {
            'conv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_inter,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1)
            },
            'bn': {
                'num_features': CH_inter
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        }
    },
    'decoder': {
        'layer_1': {
            'tconv': {
                'in_channels': CH_inter,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1),
                'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_2': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),   # (2, 2)
                'padding': (2, 1),
                'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_3': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_4': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 2),
                'padding': (2, 1),
                'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_5': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 1),
                'padding': (2, 1),
                'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_6': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (5, 3),
                'stride': (2, 2),
                'padding': (2, 1),
                'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_7': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic * 2,
                'kernel_size': (7, 5),
                'stride': (2, 1),
                'padding': (3, 2),
                # 'output_padding': (1, 0)
            },
            'bn': {
                'num_features': CH_basic * 4
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_8': {
            'tconv': {
                'in_channels': CH_basic * 4,
                'out_channels': CH_basic,
                'kernel_size': (7, 5),
                'stride': (2, 2),
                'padding': (3, 2),
                'output_padding': (0, 1)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_9': {
            'tconv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_basic,
                'kernel_size': (1, 7),
                'stride': (1, 1),
                'padding': (0, 3)
            },
            'bn': {
                'num_features': CH_basic * 2
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
        'layer_10': {
            'tconv': {
                'in_channels': CH_basic * 2,
                'out_channels': CH_input,
                'kernel_size': (7, 1),
                'stride': (1, 1),
                'padding': (3, 0)
            },
            'bn': {
                'num_features': CH_basic * 2  # not used
            },
            'act': {
                'negative_slope': 0.01,
                'inplace': True
            }
        },
    }
}
