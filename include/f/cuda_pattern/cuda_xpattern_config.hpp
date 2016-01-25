#ifndef OFJWRRJDCYLTQBNXVHHGHCFXTEVJQUACWMCUMEYYAFRSUGESCXBECIPLHVTVFFECXFCWGGSCF
#define OFJWRRJDCYLTQBNXVHHGHCFXTEVJQUACWMCUMEYYAFRSUGESCXBECIPLHVTVFFECXFCWGGSCF

#include <f/host/cuda/cuda.hpp>
#include <f/pattern/xpattern.hpp>

#include <complex>
#include <vector>

namespace f
{
    struct cuda_xpattern_config
    {
        typedef unsigned long               size_type;
        typedef double                      value_type;
        typedef std::vector<value_type>     array_type;

        int                                 device_id;
        value_type                          thickness;
        size_type                           max_dim;
        size_type                           tilt_size;
        size_type                           ug_size;
        size_type                           column_index;

        cuda_xpattern_config() = delete;

        cuda_xpattern_config( int device_id_ ) : device_id( device_id_ ) { }

        cuda_xpattern_config( cuda_xpattern_config const& config ) : device_id( config.device_id ), thickness( config.thickness ), max_dim( config.max_dim ),
                                                                     tilt_size( config.tilt_size ), ug_size( config.ug_size ), column_index( config.column_index ){}
        ~cuda_xpattern_config() { }
    };//cuda_xpattern_config

    inline cuda_xpattern_config const make_cuda_xpattern_config( xpattern<double> const& pat, int device_id )
    {
        cuda_xpattern_config config{device_id};

        config.device_id = device_id;

        config.thickness = std::imag( pat.thickness );

        config.max_dim = 0;
        for ( unsigned long index = 0; index != pat.diag.size(); ++index )
            if ( config.max_dim < (pat.diag[index]).size() )
                config.max_dim = pat.diag[index].size();

        config.tilt_size = pat.tilt_size;

        config.ug_size = pat.ug_size;

        config.column_index = pat.column_index;

        return config;
    }

}//namespace f

#endif//OFJWRRJDCYLTQBNXVHHGHCFXTEVJQUACWMCUMEYYAFRSUGESCXBECIPLHVTVFFECXFCWGGSCF

