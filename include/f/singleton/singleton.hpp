#ifndef MSINGLETON_HPP_INCLUDED_ODISFJ948ILDFJOIUIRFGDUISOIURKLJFLKJASLDKJOIUSDLKJSALKFJEOIUJSODIFUEROIUSFDLKJROIUSFDLKJF
#define MSINGLETON_HPP_INCLUDED_ODISFJ948ILDFJOIUIRFGDUISOIURKLJFLKJASLDKJOIUSDLKJSALKFJEOIUJSODIFUEROIUSFDLKJROIUSFDLKJF

namespace f
{
    template< typename T >
    struct singleton
    {
            typedef T value_type;
            typedef singleton self_type;

        private:
            struct constuctor
            {
                constuctor() { self_type::instance(); }
                inline void null_action() const { }
            };

            static constuctor constuctor_;

        public:
            static value_type& instance()
            {
                static value_type instance_;
                constuctor_.null_action();
                return instance_;
            }

        private:
            singleton( const self_type& );
            self_type& operator = ( const self_type& );
            singleton();
    };

    template<typename T>
    typename singleton<T>::constuctor singleton<T>::constuctor_;

}//namespace f

#endif//_SINGLETON_HPP_INCLUDED_ODISFJ948ILDFJOIUIRFGDUISOIURKLJFLKJASLDKJOIUSDLKJSALKFJEOIUJSODIFUEROIUSFDLKJROIUSFDLKJF

