#ifndef VAGFGEWHJAAUAABHGLJBRSUJNISWUOOUNBMSKJBWGJUUSMEXAQXQFJPBMESNTMGMTQYFGOCIL
#define VAGFGEWHJAAUAABHGLJBRSUJNISWUOOUNBMSKJBWGJUUSMEXAQXQFJPBMESNTMGMTQYFGOCIL

#include <complex>
#include <cstddef>

namespace f
{

    template< typename T, typename Zen >
    struct pattern_configer
    {
        typedef Zen                                         zen_type;
        typedef T                                           value_type;
        typedef std::complex<value_type>                    complex_type;
        typedef std::size_t                                 size_type;

        complex_type                                        ipit_;
        size_type                                           column_index_;
        size_type                                           ug_size_;

        void dump_pattern_configer() const
        {
            std::cout << "\ndump pattern_configer:";
            std::cout << "\nipit:\t " << ipit_;
            std::cout << "\ncolumn_index:\t " << column_index_;
            std::cout << "\nug_size:\t" << ug_size_ << std::endl;
        }

        //interface
        complex_type const ipit() const
        {
            return ipit_;
        }

        size_type column_index() const
        {
            return column_index_;
        }

        size_type ug_size() const
        {
            return ug_size_;
        }

        //setter
        complex_type& ipit()
        {
            return ipit_;
        }

        //used as initializer/proposed usage:
        //
        //      our_dynamic_solver ods;
        //      ods.config_ipit( 1.0  );
        //      ods.config_column_index(0);
        //
        void config_ipit( value_type x )
        {
            ipit_ = complex_type{ value_type{0}, x };
        }

        void config_ipit( complex_type const& ipit__ )
        {
            ipit_ = ipit__;
        }

        void config_column_index( size_type column_index__ )
        {
            column_index_ = column_index__;
        }

        void config_ug_size( size_type ug_size__ )
        {
            ug_size_ = ug_size__;
        }

    };

}//namespace f

#endif//VAGFGEWHJAAUAABHGLJBRSUJNISWUOOUNBMSKJBWGJUUSMEXAQXQFJPBMESNTMGMTQYFGOCIL

