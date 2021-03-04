basic_channels = 16
k_G = 32
s = 2
p_G = int((k_G - 2) / s)
k_D = 31
p_D = int((k_D - 1) / s)
out_padding = 0

channels = [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
pars_generator = {
    'encoder': {
        'layer1': {
            'conv': {
                'in_channels': channels[0],
                'out_channels': channels[1],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[1],
            },
        },
        'layer2': {
            'conv': {
                'in_channels': channels[1],
                'out_channels': channels[2],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[2],
            },
        },
        'layer3': {
            'conv': {
                'in_channels': channels[2],
                'out_channels': channels[3],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[3],
            },
        },
        'layer4': {
            'conv': {
                'in_channels': channels[3],
                'out_channels': channels[4],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[4],
            },
        },
        'layer5': {
            'conv': {
                'in_channels': channels[4],
                'out_channels': channels[5],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[5],
            },
        },
        'layer6': {
            'conv': {
                'in_channels': channels[5],
                'out_channels': channels[6],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[6],
            },
        },
        'layer7': {
            'conv': {
                'in_channels': channels[6],
                'out_channels': channels[7],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[7],
            },
        },
        'layer8': {
            'conv': {
                'in_channels': channels[7],
                'out_channels': channels[8],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[8],
            },
        },
        'layer9': {
            'conv': {
                'in_channels': channels[8],
                'out_channels': channels[9],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[9],
            },
        },
        'layer10': {
            'conv': {
                'in_channels': channels[9],
                'out_channels': channels[10],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[10],
            },
        },
        'layer11': {
            'conv': {
                'in_channels': channels[10],
                'out_channels': channels[11],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G
            },
            'norm': {
                'num_features': channels[11],
            },
        }
    },
    'decoder': {
        'layer1': {
            'conv': {
                'in_channels': channels[11],  # channels[11] * 2
                'out_channels': channels[10],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[10] * 2,
            },
            'norm_before': {
                'num_features': channels[10],
            },
        },
        'layer2': {
            'conv': {
                'in_channels': channels[10] * 2,
                'out_channels': channels[9],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[9] * 2,
            },
            'norm_before': {
                'num_features': channels[9],
            },
        },
        'layer3': {
            'conv': {
                'in_channels': channels[9] * 2,
                'out_channels': channels[8],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[8] * 2,
            },
            'norm_before': {
                'num_features': channels[8],
            },
        },
        'layer4': {
            'conv': {
                'in_channels': channels[8] * 2,
                'out_channels': channels[7],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[7] * 2,
            },
            'norm_before': {
                'num_features': channels[7],
            },
        },
        'layer5': {
            'conv': {
                'in_channels': channels[7] * 2,
                'out_channels': channels[6],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[6] * 2,
            },
            'norm_before': {
                'num_features': channels[6],
            },
        },
        'layer6': {
            'conv': {
                'in_channels': channels[6] * 2,
                'out_channels': channels[5],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[5] * 2,
            },
            'norm_before': {
                'num_features': channels[5],
            },
        },
        'layer7': {
            'conv': {
                'in_channels': channels[5] * 2,
                'out_channels': channels[4],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[4] * 2,
            },
            'norm_before': {
                'num_features': channels[4],
            },
        },
        'layer8': {
            'conv': {
                'in_channels': channels[4] * 2,
                'out_channels': channels[3],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[3] * 2,
            },
            'norm_before': {
                'num_features': channels[3],
            },
        },
        'layer9': {
            'conv': {
                'in_channels': channels[3] * 2,
                'out_channels': channels[2],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[2] * 2,
            },
            'norm_before': {
                'num_features': channels[2],
            },
        },
        'layer10': {
            'conv': {
                'in_channels': channels[2] * 2,
                'out_channels': channels[1],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[1] * 2,
            },
            'norm_before': {
                'num_features': channels[1],
            },
        },
        'layer11': {
            'conv': {
                'in_channels': channels[1] * 2,
                'out_channels': channels[0],
                'kernel_size': k_G,
                'stride': s,
                'padding': p_G,
                'output_padding': out_padding
            },
            'norm': {
                'num_features': channels[0] * 2,
            },
            'norm_before': {
                'num_features': channels[0],
            },
        },
    }
}

pars_discriminator = {
    'layers': {
        'layer1': {
            'conv': {
                'in_channels': 2,
                'out_channels': channels[1],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D,
            },
            'vbn': {
                'num_features': channels[1],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer2': {
            'conv': {
                'in_channels': channels[1],
                'out_channels': channels[2],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D,
            },
            'vbn': {
                'num_features': channels[2],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer3': {
            'conv': {
                'in_channels': channels[2],
                'out_channels': channels[3],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D,
            },
            'vbn': {
                'num_features': channels[3],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer4': {
            'conv': {
                'in_channels': channels[3],
                'out_channels': channels[4],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D,
            },
            'vbn': {
                'num_features': channels[4],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer5': {
            'conv': {
                'in_channels': channels[4],
                'out_channels': channels[5],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D,
            },
            'vbn': {
                'num_features': channels[5],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer6': {
            'conv': {
                'in_channels': channels[5],
                'out_channels': channels[6],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D,
            },
            'vbn': {
                'num_features': channels[6],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer7': {
            'conv': {
                'in_channels': channels[6],
                'out_channels': channels[7],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D
            },
            'vbn': {
                'num_features': channels[7],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer8': {
            'conv': {
                'in_channels': channels[7],
                'out_channels': channels[8],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D
            },
            'vbn': {
                'num_features': channels[8],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer9': {
            'conv': {
                'in_channels': channels[8],
                'out_channels': channels[9],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D
            },
            'vbn': {
                'num_features': channels[9],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer10': {
            'conv': {
                'in_channels': channels[9],
                'out_channels': channels[10],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D
            },
            'vbn': {
                'num_features': channels[10],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
        'layer11': {
            'conv': {
                'in_channels': channels[10],
                'out_channels': channels[11],
                'kernel_size': k_D,
                'stride': s,
                'padding': p_D
            },
            'vbn': {
                'num_features': channels[11],
            },
            'act': {
                'negative_slope': 0.3
            }
        },
    },
    'squeeze_conv': {
                'in_channels': channels[11],
                'out_channels': 1,
                'kernel_size': 1,
    },
    'lrelu': {

    },
    'project': {
        'in_features': 8,
        'out_features': 1
    }
}


