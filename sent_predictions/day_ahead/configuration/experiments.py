Layer_name_list = ['conv',
                   'conv_3d',
                   'time_distr_conv',
                   'time_distr_conv_3d',
                   'lstm',
                   'hidden_dense',
                   'dense',
                   'Flatten',
                   'Dropout']

fuzzy1 = {'input': [
    ('dense', {1})
],
    'output': [
        ('dense', {1})
    ],
}
cnn1 = {'input': [('conv', [2, 4]),
                  ('Flatten', []),
                  ('dense', {0.5, 1, 0.25}),
                  # ('Dropout', [0.75])
                  # ('dense', {64, 128, 256})
                  ],
        'data_row': [('dense', {4, 2}),
                     # ('dense', {64, 360, 720}),
                     # ('dense', {64, 128})
                     ],
        'output': [('dense', {0.5, 0.25}),
                   # ('Dropout', [0.75]),
                   # ('dense', {'linear', 2520, 3960}),
                   ('dense', {32, 64, 128})
                   ],
        }
cnn2 = {'input': [('conv', [2, 4]),
                  ('conv', [2, 4]),
                  ('Flatten', []),
                  ('dense', {0.5, 0.25}),
                  ('dense', {0.5, 0.25})
                  ],
        'data_row': [('dense', {2, 3, 4}),
                     # ('dense', {64, 360, 720}),
                     # ('dense', {64, 128})
                     ],
        'output': [('dense', {0.5, 0.25}),
                   ('dense', {'linear', 0.5, 0.25}),
                   ('dense', {64, 32, 128})
                   ],
        }
cnn3 = {'input': [('conv', [2, 4]),
                  ('conv', [2, 4]),
                  ('Flatten', []),
                  ('dense', {0.5, 0.25}),
                  ('dense', {0.5, 0.25}),
                  ('dense', {64, 12, 128})
                  ],
        'data_row': [('dense', {2, 3}),
                     ('dense', {3, 2, 0.5}),
                     ],
        'output': [('dense', {0.5, 0.25}),
                   ('dense', {0.5, 0.25}),
                   ('dense', {0.5, 0.25}),
                   ('dense', {64, 128, 12})
                   ],
        }
mlp1 = {'input': [('dense', {1, 2}),
                  # ('dense', {'linear'}),
                  # ('dense', {28})
                  ],
        'output': [('dense', {1, 0.5, 2}),
                   # ('dense', {'linear'}),
                   # ('dense', {64})
                   ],
        }
mlp2 = {'input': [('dense', [4, 8]),
                  # ('dense', {720, 1680, 3960}),
                  ('dense', {64, 128, 256})
                  ],
        'output': [('dense', {1, 2, 3, 1024}),
                   # ('dense', {'linear', 2520, 360}),
                   ('dense', {64, 128, 256})
                   ],
        }
mlp3 = {'input': [('dense', [4, 8]),
                  # ('dense', {720, 128}),
                  ('dense', {720, 2, 0.5, 256}),
                  ('dense', {64, 128, 256})
                  ],
        'output': [('dense', {1, 2}),
                   ('dense', {'linear', 2, 0.5, 720}),
                   # ('dense', {0.5, 720, 1680}),
                   ('dense', {64, 128, 256})
                   ],
        }

distributed_cnn1 = {'input': [('conv', 2),
                              ('Flatten', []),
                              ('dense', 0.5),
                              # ('Dropout', [0.75])
                              # ('dense', {64, 128, 256})
                              ],
                    'data_row': [('dense', 4),
                                 # ('dense', {64, 360, 720}),
                                 # ('dense', {64, 128})
                                 ],
                    'output': [('dense', 0.5),
                               # ('Dropout', [0.75]),
                               # ('dense', {'linear', 2520, 3960}),
                               ('dense', 32)
                               ],
                    }

distributed_mlp1 = {'input': [('dense', 1),
                              # ('dense', {'linear'}),
                              # ('dense', {28})
                              ],
                    'output': [('dense', 1),
                               # ('dense', {'linear'}),
                               # ('dense', {64})
                               ],
                    }
distributed_mlp2 = {'input': [('dense', 2),
                              # ('dense', {'linear'}),
                              ('dense', 0.5)
                              ],
                    'output': [('dense', 1),
                               # ('dense', {'linear'}),
                               ('dense', 64)
                               ],
                    }

lstm1 = {'input': [('lstm', {0.5, 1, 2}),
                   ('lstm', {0.5, 1}),
                   ('Flatten', []),
                   ('dense', {1, 0.5, 2})],
         'output': [('dense', {1, 0.5}),
                    ('dense', {12, 64, 128})]
         }
lstm2 = {'input': [('lstm', {0.5, 1, 2}),
                   ('Flatten', []),
                   ('dense', {1, 3, 2}),
                   ('Reshape', []),  #: [32, 32]}
                   ('lstm', {0.5, 1, 2}),
                   ('Flatten', []),
                   ('dense', {1, 0.5, 2})
                   ],
         'output': [('dense', {12, 64, 128})
                    ]
         }
lstm3 = {'input': [('lstm', {0.5, 1, 2}),
                   ('Flatten', []),
                   ('dense', {1, 3, 2}),
                   ('Reshape', []),
                   ('lstm', {0.5, 1, 2}),
                   ('Flatten', []),
                   ('dense', {1, 3, 2}),
                   ('Reshape', []),
                   ('lstm', {0.5, 1}),
                   ('Flatten', []),
                   ('dense', {1, 2, 0.5})
                   ],
         'output': [('dense', {1, 0.5, 0.25}),
                    ('dense', {12, 64, 128}),
                    ]
         }
lstm4 = {'input': [('lstm', {0.5, 1, 2}),
                   ('Flatten', []),
                   ('dense', {1, 0.5}),
                   ('dense', {1, 0.5, 0.25})],
         'output': [('dense', {12, 64, 128})
                    ]
         }

distributed_lstm1 = {'input': [('lstm', 1),
                               ('Flatten', []),
                               ('dense', 0.25),
                               ('dense', 0.5),
                               ],
                     'output': [('dense', 0.25),
                                ('dense', 0.5),
                                ('dense', 32)
                                ],
                     }
